import os
import torch
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np

args = parse()

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

    server_params = {n:p for n, p in server_model.named_parameters() if p.requires_grad == True}
    if server_opt == 'sgd':
        server_opt = torch.optim.SGD(server_params.values(), lr=server_lr)
    elif server_opt == 'adam':
        server_opt = torch.optim.AdamW(server_params.values(), lr=server_lr)
    else:
        raise ValueError()
    server_opt.zero_grad()
    
    shuffled_data = {}
    for ep in range(args.num_epochs):
        print(f"\n=== Epoch {ep+1}/{args.num_epochs} ===")
        num_comm_rounds = len(list(client_dataloaders[0])) // (args.local_iter_per_round * num_gpus)
        pbar = tqdm(range(num_comm_rounds), desc=f"Epoch {ep+1}")
        flag = [0]*args.clients # for dataset shuffling
        for rnd in pbar:
            aggregate = None
            client_ids = np.arange(args.clients) #torch.randperm(len(client_dataloaders))[:server_batch] # here is full participation
            for i,client_id in enumerate(client_ids):
                
                client_model = deepcopy(server_model)
                client_model.config.use_cache = False  # Disable caching for training
                # Local Training
                client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
                client_opt.zero_grad()
                client_loader = client_dataloaders[client_id]
                client_model, client_opt = accelerator.prepare(client_model, client_opt)

                sketching_mat = {}
                mask_set = {}
                m = m_list[client_id] # the effective rank of client i
                print('ratio:', r/m)
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
                if rnd == 0 and flag[client_id] == 0:
                    all_batches = list(client_loader) # every process will get the full dataset
                    seed = base_seed + ep * 10000 + client_id * 100 + accelerator.process_index * 10
                    random.seed(seed)
                    random.shuffle(all_batches)

                    # Each process takes its own chunk
                    process_chunk_size = len(all_batches) // accelerator.num_processes
                    start = accelerator.process_index * process_chunk_size
                    end = start + process_chunk_size if accelerator.process_index < accelerator.num_processes - 1 else len(all_batches)
                    shuffled_data[client_id] = all_batches[start:end]
                    flag[client_id] = 1

                # Training on process-specific chunk
                # Slice the data chunk for this communication round
                # Each client sees local_iter_per_round * mini-batches each round
                start_idx = rnd * args.local_iter_per_round
                end_idx   = min((rnd + 1) * args.local_iter_per_round, len(shuffled_data[client_id]))
                local_data = shuffled_data[client_id][start_idx:end_idx]
                for step, batch in enumerate(local_data):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    #loss.backward()
                    accelerator.backward(loss)
                    #accelerator.print(f"current epoch: {rnd +1}, client: {i} size: {input_ids.shape}")
                    
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
                    #neg_client_delta[n] = accelerator.gather(neg_client_delta[n])
                    neg_client_delta[n] = accelerator.reduce(neg_client_delta[n], reduction="mean")
                # Aggregation
                if aggregate is None:
                    aggregate = neg_client_delta
                else:
                    for n, delta in neg_client_delta.items():
                        aggregate[n] += delta
                torch.cuda.empty_cache()
                accelerator.free_memory()  # Releases memory allocated by Accelerate *** IMPORTANT, Memory explosion
            
            # Server model update
            server_params = {normalize_name(k): v for k, v in server_params.items()}  # for parallel, test
            for n, sp in server_params.items():
                sp.grad = aggregate[n] / args.clients
            server_opt.step()
            server_opt.zero_grad()

            # Synchronize processes
            accelerator.wait_for_everyone()

        if accelerator.is_main_process: 
            save_path = "..."
            os.makedirs(save_path, exist_ok=True)
            server_model.save_pretrained(save_path)
