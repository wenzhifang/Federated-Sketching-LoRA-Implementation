import os
import torch
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np

import models

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

def delta_average(client_model, server_params, norm_sum, aggregate, lora_rank, accelerator):
    device = accelerator.device
    if aggregate == None:
        aggregate = {}
        norm_sum = {}
    client_params = {normalize_name(n): cp for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    
    for name, param in client_params.items():
        param = accelerator.reduce(param, reduction="mean")
        if "lora_B" in name or "lora_A" in name:
            client_rank = param.shape[1] if "lora_B" in name else param.shape[0]
            if "lora_B" in name:
                padded_param = torch.zeros(param.data.shape[0], lora_rank, device=device)
                padded_param[:, :client_rank] = server_params[name][:, :client_rank].detach() - param.detach()
            elif "lora_A" in name:
                padded_param = torch.zeros(lora_rank, param.data.shape[1], device=device)
                padded_param[:client_rank, :] = server_params[name][:client_rank, :].detach() -  param.detach()
          
        else:
            padded_param = server_params[name].detach() - param.detach()
        
        if name not in aggregate:
            aggregate[name] = padded_param.clone()
            norm_sum[name] = 1
        else: 
            #print(aggregate[name].shape, weighted_sum.shape)
            aggregate[name] = aggregate[name] + padded_param
            norm_sum[name] += 1 
    
    return norm_sum, aggregate


def fl_cmu_lora_train_het(server_model, fronzen_model, client_dataloaders, server_opt, server_lr, client_lr, eval_freq, r, m_list, accelerator, base_seed):

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
            norm_sum = None
            client_ids = torch.randperm(args.clients)[:args.server_batch]
                        
            for i,client_id in enumerate(client_ids):
                
                client_model_base = deepcopy(fronzen_model)
                client_model = models.add_adapter(client_model_base, m_list[client_id], args.lora_alpha)
                # initialize client model with server params    
                client_params = {n: cp for n, cp in client_model.named_parameters() if cp.requires_grad==True}
            
                for n, sp in server_params.items():
                    if "lora_B" in n: # LoRA module: truncate server's weights for the client
                        _, r_s = sp.shape  # Server rank
                        r_c = m_list[client_id]  # Client rank
                        if r_c <= r_s:
                            client_params[n].data = sp[:, :r_c].clone()
                        else:
                            raise ValueError(f"Client LoRA rank ({r_c}) exceeds server LoRA rank ({r_s}).")
                    elif "lora_A" in n:
                        r_s, _ = sp.shape  # Server rank
                        r_c = m_list[client_id]  # Client rank
                        if r_c <= r_s:
                            client_params[n].data = sp[:r_c, :].clone()
                        else:
                            raise ValueError(f"Client LoRA rank ({r_c}) exceeds server LoRA rank ({r_s}).")
                    else:
                        # Final linear layer or other trainable parameters: directly copy
                        client_params[n].data = sp.clone()

                client_model.config.use_cache = False  # Disable caching for training
                client_model = optimize_model_memory(client_model) # set train mode and set requires_grad=True
                # Local Training
                #client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
                client_opt = torch.optim.AdamW(client_model.parameters(), lr=client_lr)
                client_opt.zero_grad()
                client_loader = client_dataloaders[client_id]
                client_model, client_opt, client_loader = accelerator.prepare(client_model, client_opt, client_loader)
                
                # one communication round // H=20    
                train_data_list = list(client_loader)
                for step in range(args.local_iter_per_round):
                    batch = random.choice(train_data_list)
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    
                    client_opt.step()
                    client_opt.zero_grad()

                norm_sum, aggregate = delta_average(client_model, server_params, norm_sum, aggregate, r, accelerator) 

                torch.cuda.empty_cache()
                accelerator.free_memory()
            server_opt.zero_grad()
            for n, sp in server_params.items():
                sp.grad = aggregate[n] / norm_sum[n]
            server_opt.step()
            server_opt.zero_grad()
            
            accelerator.wait_for_everyone()

        if accelerator.is_main_process: 
            save_path = f"./model_parameters_set/HetLora{args.lora_r}_H{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{ep}"
            os.makedirs(save_path, exist_ok=True)
            server_model.save_pretrained(save_path)
        
