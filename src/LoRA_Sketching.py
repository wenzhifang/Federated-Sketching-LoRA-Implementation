import os
import torch
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np

args = parse()

def optimize_model_memory(model):
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()
    return model

def normalize_name(name):
    while name.startswith("module."):
        name = name[len("module."):]  # Remove module. until no longer present
    return name

def fl_slora_train_llama_het(server_model, client_dataloaders, server_opt, server_lr, client_lr, eval_freq, r, m_list, accelerator, base_seed):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available.")

    print(f"Using device: {accelerator.device}")
    print(f"Number of processes: {accelerator.state.num_processes}")
    print(f"Process index: {accelerator.state.local_process_index}")
    
    server_model.to(accelerator.device) #required if don't call quantized model via bitsandbytes
    
    server_params = {n:p for n, p in server_model.named_parameters() if p.requires_grad == True}
    if server_opt == 'sgd':
        server_opt = torch.optim.SGD(server_params.values(), lr=server_lr)
    elif server_opt == 'adam':
        server_opt = torch.optim.AdamW(server_params.values(), lr=server_lr)
    else:
        raise ValueError()
    server_opt.zero_grad()
    
    for ep in range(args.num_comm_rounds//eval_freq):
        print(f"\n=== Saving round {ep+1}/{args.num_comm_rounds//eval_freq} ===")
        pbar = tqdm(range(eval_freq), desc=f"Epoch {ep+1}")
        for rnd in pbar:
            aggregate = None

            client_ids = torch.randperm(args.clients)[:args.server_batch]
            for i, client_id in enumerate(client_ids):
                
                client_model = deepcopy(server_model)
                client_model.config.use_cache = False  # Disable caching for training
                client_model = optimize_model_memory(client_model) # set train mode and set requires_grad=True
                # Local Training
                # client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
                client_opt = torch.optim.AdamW(client_model.parameters(), lr=client_lr)
                client_opt.zero_grad()
                client_loader = client_dataloaders[client_id]
                client_model, client_opt, client_loader = accelerator.prepare(client_model, client_opt, client_loader)

                # We provided a mathematically equivalent implementation based on the existing PEFT package through some algebra operations, which is enough to validate the performance of FSLoRA
                # Note that a more computation- and memory-efficient implementation involves directly extracting local LoRA modules from the global LoRA via sketching, followed by tuning the 
                #   smaller local modules. However, this approach requires explicitly incorporating the sketching ratio r/k_i into the local LoRA matrices B_i and A_i, which entails modifying 
                #   the PEFT package to accept r/k_i as an additional input.
                # We adopt the algebraic transformation method here for simplicity and to facilitate reproducibility.
            
                sketching_mat = {}
                mask_set = {}
                m = m_list[client_id] # the effective rank of client i
                #print('ratio:', r/m)
                for n,p in client_model.named_parameters():
                    S = torch.ones_like(p.data) # for the final linear layer
                    mask = torch.ones_like(p.data)
                    
                    rand_perm = torch.randperm(r)[:m]
                    if 'lora_B' == n:
                        S = torch.zeros_like(p.data)
                        mask = torch.zeros_like(p.data)
                        mask[:, rand_perm] = 1
                        S[:, rand_perm] = r / m
                        p.data *= S ## (BS) * A,  p.data = BS 
                    elif 'lora_A' == n:
                        S = torch.zeros_like(p.data)
                        mask = torch.zeros_like(p.data)
                        mask[rand_perm, :] = 1
                        S[rand_perm, :] = r / m
                    sketching_mat[n] = S
                    mask_set[normalize_name(n)] = mask # for parallel, test
                    
                train_data_list = list(client_loader)
                for step in range(args.local_iter_per_round):
                    batch = random.choice(train_data_list)
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    
                    for n,p in client_model.named_parameters():
                        if 'lora_B' == n:
                            p.grad *= sketching_mat[n]**2

                    client_opt.step()
                    client_opt.zero_grad()
                        
                for n,p in client_model.named_parameters():
                    if 'lora_B' == n:
                        epsilon = 1e-8  # A small constant to prevent division by zero
                        p.data /= (sketching_mat[n] + epsilon) ## recover B from BS    
                # The above update for B involves a algebra transformation, in terms of scaled gradient
                
                neg_client_delta = {normalize_name(n): (server_params[normalize_name(n)].data - cp.data)*mask_set[normalize_name(n)] for n,cp 
                                    in client_model.named_parameters() if cp.requires_grad} # for parallel, test
              
                for n in neg_client_delta:
                    neg_client_delta[n] = accelerator.reduce(neg_client_delta[n], reduction="mean")
                # Aggregation
                if aggregate is None:
                    aggregate = neg_client_delta
                else:
                    for n, delta in neg_client_delta.items():
                        aggregate[n] += delta
                del client_model
                torch.cuda.empty_cache()
                accelerator.free_memory()  # Releases memory allocated by Accelerate *** IMPORTANT, Memory explosion
            
            # Server model update
            server_params = {normalize_name(k): v for k, v in server_params.items()}
            for n, sp in server_params.items():
                sp.grad = aggregate[n]/args.server_batch
            server_opt.step()
            server_opt.zero_grad()
            accelerator.wait_for_everyone()

        if accelerator.is_main_process: 
            save_path = f"./model_parameters_set/slora{args.lora_r}_H{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{ep}"
            os.makedirs(save_path, exist_ok=True)
            server_model.save_pretrained(save_path)
