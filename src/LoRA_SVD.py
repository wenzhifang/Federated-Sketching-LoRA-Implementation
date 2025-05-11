import os
import torch
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np
import math

import models

args = parse()

def normalize_name(name):
    while name.startswith("module."):
        name = name[len("module."):]  # Remove module. until no longer present
    return name

def weight_aggregation(client_model, aggregate, accelerator, r):
    if aggregate == None:
        aggregate = {}
    client_params = {normalize_name(n): cp for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    for name, param in client_params.items():
        client_params[name] = accelerator.reduce(param, reduction="mean")
    
    for name, param in client_params.items():
        if "lora_B" in name:
            paired_param_name = name.replace("lora_B", "lora_A")
            delta_W = torch.matmul(param.detach(), client_params[paired_param_name].detach())
            aggregate[name] = aggregate.get(name, torch.zeros_like(delta_W)) + (1/math.sqrt(r))*delta_W.detach()
            #aggregate[paired_param_name] = aggregate[name].clone()
        if "lora_B" not in name and "lora_A" not in name:
            aggregate[name] = aggregate.get(name, torch.zeros_like(param)) + param.detach()             
            
    return aggregate

def local_model_initialization(server_params, aggregate, num_client):
    U_set, S_set, V_set = {}, {}, {}
    for name, param in aggregate.items():
        if "lora_B" in name:
            U, S, V = torch.linalg.svd(param/num_client) # actually V is Vh USV^H = param, here V represent Vh
            U, S, V = U.detach(), S.detach(), V.detach()
            U_set[name] = U
            S_set[name] = S
            paired_param_name = name.replace("lora_B", "lora_A")
            V_set[paired_param_name] = V
            _, r = server_params[name].shape
            with torch.no_grad():
                server_params[name].data.copy_(torch.matmul(U[:, :r], torch.diag(S[:r])*math.sqrt(r)))
                server_params[paired_param_name].data.copy_(V[:r,:])
            
        if "lora_B" not in name and "lora_A" not in name:
            with torch.no_grad():
                U_set[name] = (param/num_client).detach()
                server_params[name].data.copy_((param/num_client))
            
    return U_set, S_set, V_set

def fl_svd_lora_train_het(server_model, fronzen_model, client_dataloaders, client_lr, eval_freq, m_list, accelerator):
    server_params = {n:p for n, p in server_model.named_parameters() if p.requires_grad == True}
    
    # SVD for initialization
    # U_set, S_set, V_set = svd_reconstruct(server_params, accelerator) 
    torch.cuda.empty_cache()
    accelerator.free_memory()
    for ep in range(args.num_comm_rounds//eval_freq):
        print(f"\n=== Saving round {ep+1}/{args.num_comm_rounds//eval_freq} ===")
        pbar = tqdm(range(eval_freq), desc=f"Epoch {ep+1}")
        for rnd in pbar:
            aggregate = None
            client_ids = torch.randperm(len(client_dataloaders))[:args.server_batch]
            for i,client_id in enumerate(client_ids):
                
                client_model_base = deepcopy(fronzen_model) #client_model_base is necessary as client_model_base will be updated in place. the return value of PEFT just give you more convenient features
                client_model = models.add_adapter(client_model_base, m_list[client_id], args.lora_alpha)
                # initialize client model with server params    
                client_params = {n: cp for n, cp in client_model.named_parameters() if cp.requires_grad==True}
                r_c = m_list[client_id]  # Client rank
                if ep or rnd:
                    for n, cp in client_params.items():
                        if "lora_B" in n: # LoRA module: truncate server's weights for the client
                            B = torch.matmul(U_set[n][:, :r_c].detach(), torch.diag(S_set[n][:r_c].detach()*math.sqrt(r_c))) # r_c is the scaling factor 1/s
                            if cp.data.shape == B.shape:
                                cp.data = B
                            else:
                                raise ValueError(f"svd mismatch B.")
                        elif "lora_A" in n:
                            A = V_set[n][:r_c,:].clone()
                            if cp.data.shape == A.shape:
                                cp.data = A
                            else:
                                print(cp.data.shape, A.shape)
                                raise ValueError(f"svd mismatch A.")
                        else:
                            # Final linear layer or other trainable parameters: directly copy
                            cp.data = U_set[n].clone()
                
                client_model.config.use_cache = False  # Disable caching for training
                
                client_opt = torch.optim.AdamW(client_params.values(), lr=client_lr)
                #client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
                client_opt.zero_grad()
                client_loader = client_dataloaders[client_id]
                client_model, client_opt, client_loader = accelerator.prepare(client_model, client_opt, client_loader)
    
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
                aggregate = weight_aggregation(client_model, aggregate, accelerator, r_c) 

                del client_model
                del client_opt
                torch.cuda.empty_cache()
                accelerator.free_memory()

            U_set, S_set, V_set = local_model_initialization(server_params, aggregate, args.server_batch) # server_model will be updated inside
            torch.cuda.empty_cache()
            accelerator.free_memory()
            accelerator.wait_for_everyone()

        if accelerator.is_main_process: 
            save_path = f"./model_parameters_set/FlexLoRA{args.lora_r}_H{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{ep}"
            os.makedirs(save_path, exist_ok=True)
            server_model.save_pretrained(save_path)
        
