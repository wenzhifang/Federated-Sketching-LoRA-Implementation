# For CMU Cho ====== heterogeneout rank aggregation 
def weighted_average(client_model, norm_sum, aggregate, lora_rank, device):
    if aggregate == None:
        aggregate = {}
        norm_sum = {}
    client_params = {n: cp.to(device) for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    
    for name, param in client_params.items():
        if "lora_B" in name or "lora_A" in name:
            client_rank = param.shape[1] if "lora_B" in name else param.shape[0]
            if "lora_B" in name:
                weighted_sum = torch.zeros(param.data.shape[0], lora_rank, device=device)
                padded_param = torch.zeros(param.data.shape[0], lora_rank, device=device)
                padded_param[:, :client_rank] = param.detach()
            elif "lora_A" in name:
                weighted_sum = torch.zeros(lora_rank, param.data.shape[1], device=device)
                padded_param = torch.zeros(lora_rank, param.data.shape[1], device=device)
                padded_param[:client_rank, :] = param.detach()
            paired_param_name = name.replace("lora_B","lora_A") if "lora_B" in name else name.replace("lora_A","lora_B")
            paired_param = client_params[paired_param_name]
            frob_norm = torch.norm(torch.matmul(param.detach(), paired_param.detach()), p="fro").item()            
            weighted_sum = frob_norm * padded_param.detach()
        else:
            frob_norm = 1  
            weighted_sum = param.detach()
        
        if name not in aggregate:
            aggregate[name] = weighted_sum.clone()
            norm_sum[name] = torch.tensor(frob_norm, device=device)  
        else: 
            aggregate[name] = aggregate[name] + weighted_sum
            norm_sum[name] += frob_norm 
    
    return norm_sum, aggregate

def weighted_delta_average(client_model, server_params, norm_sum, aggregate, aggregate_weight, lora_rank, device):
    if aggregate == None:
        aggregate_weight = {}
        aggregate = {}
        norm_sum = {}
    client_params = {n: cp.to(device) for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    
    for name, param in client_params.items():
        if "lora_B" in name or "lora_A" in name:
            client_rank = param.shape[1] if "lora_B" in name else param.shape[0]
            if "lora_B" in name:
                weighted_sum = torch.zeros(param.data.shape[0], lora_rank, device=device)
                padded_param = torch.zeros(param.data.shape[0], lora_rank, device=device)
                padded_param_weight = torch.zeros(param.data.shape[0], lora_rank, device=device)
                padded_param[:, :client_rank] = server_params[name][:, :client_rank].detach() - param.detach()
                padded_param_weight[:, :client_rank] = server_params[name][:, :client_rank].detach()
            elif "lora_A" in name:
                weighted_sum = torch.zeros(lora_rank, param.data.shape[1], device=device)
                padded_param = torch.zeros(lora_rank, param.data.shape[1], device=device)
                padded_param_weight = torch.zeros(lora_rank, param.data.shape[1], device=device)
                padded_param[:client_rank, :] = server_params[name][:client_rank, :].detach() -  param.detach()
                padded_param_weight[:client_rank, :] = server_params[name][:client_rank, :].detach()
            paired_param_name = name.replace("lora_B","lora_A") if "lora_B" in name else name.replace("lora_A","lora_B")
            frob_norm = torch.norm(torch.matmul(param.detach(), client_params[paired_param_name].detach()), p="fro").item()
            weighted_sum = frob_norm * padded_param.detach()
            weighted_sum_weight = frob_norm * padded_param_weight.detach()
        else:
            frob_norm = 1  
            weighted_sum = server_params[name].detach() - param.detach()
            weighted_sum_weight = server_params[name].detach()
        
        if name not in aggregate:
            aggregate[name] = weighted_sum.clone()
            aggregate_weight[name] = weighted_sum_weight.clone()
            norm_sum[name] = torch.tensor(frob_norm, device=device)  
        else: 
            #print(aggregate[name].shape, weighted_sum.shape)
            aggregate[name] = aggregate[name] + weighted_sum
            aggregate_weight[name] = aggregate_weight[name] + weighted_sum_weight.clone()
            norm_sum[name] += frob_norm 
    
    return norm_sum, aggregate, aggregate_weight

def delta_average(client_model, server_params, norm_sum, aggregate, lora_rank, device):
    if aggregate == None:
        aggregate = {}
        norm_sum = {}
    client_params = {n: cp.to(device) for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    
    for name, param in client_params.items():
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

def weight_aggregation(client_model, aggregate, device):
    if aggregate == None:
        aggregate = {}
    client_params = {n: cp.to(device) for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    
    for name, param in client_params.items():
        if "lora_B" in name:
            paired_param_name = name.replace("lora_B", "lora_A")
            delta_W = torch.matmul(param.detach(), client_params[paired_param_name].detach())
            aggregate[name] = aggregate.get(name, torch.zeros_like(delta_W)) + delta_W.detach() 
            aggregate[paired_param_name] = aggregate[name].clone()
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
            server_params[name].data.copy_(torch.matmul(U[:, :r], torch.diag(S[:r])))
            server_params[paired_param_name].data.copy_(V[:r,:])
            
        if "lora_B" not in name and "lora_A" not in name:
            U_set[name] = (param/num_client).detach()
            server_params[name].data.copy_((param/num_client))
            
    return U_set, S_set, V_set

def svd_reconstruct(server_params):
    U_set, S_set, V_set = {}, {}, {}
    for name, param in server_params.items():
        if "lora_B" in name:
            #print(f"Parameter {name} is on device: {param.device}")
            paired_param_name = name.replace("lora_B", "lora_A")
            delta_W = torch.matmul(param.detach(), server_params[paired_param_name].detach())
            U, S, V = torch.linalg.svd(delta_W) # actually V is Vh USV^H = delta_W, here V represent Vh
            U, S, V = U.detach(), S.detach(), V.detach()
            U_set[name] = U
            S_set[name] = S
            V_set[paired_param_name] = V
        if "lora_B" not in name and "lora_A" not in name:
            U_set[name] = param.detach()     
    return U_set, S_set, V_set

def weight_aggregation_rlora(client_model, aggregate, rc, device):
    for name, module in client_model.named_modules():
        # Check the scaling factor
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            scaling = getattr(module, "scaling", 1.0)  # Default to 1.0 if not set
            #print(scaling)
            break

    if aggregate == None:
        aggregate = {}
    client_params = {n: cp.to(device) for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    for name, param in client_params.items():
        if "lora_B" in name:
            paired_param_name = name.replace("lora_B", "lora_A")
            delta_W = torch.matmul(param.detach(), client_params[paired_param_name].detach())
            aggregate[name] = aggregate.get(name, torch.zeros_like(delta_W)) + scaling['default'] * delta_W.detach() #/ torch.sqrt(torch.tensor(rc)) #torch.sqrt
            #aggregate[paired_param_name] = aggregate[name].clone()
        if "lora_B" not in name and "lora_A" not in name:
            aggregate[name] = aggregate.get(name, torch.zeros_like(param)) + param.detach()             
            
    return aggregate


def merge_lora_adapeters_into_original_model(fronzen_model, aggregate, num_client):
    state_dict = fronzen_model.state_dict()
    #for n, fp in state_dict.items():
        #if "layer.0" in n:
        #print(n)
    #for name, param in aggregate.items():
        #print(name)
    for name, param in aggregate.items():
        if "lora_B" in name:
            base_param_name = name.replace(".lora_B", "").replace(".default", "")
            state_dict[base_param_name].data.add_(param / num_client)
        if "lora_B" not in name and "lora_A" not in name:
            base_param_name = name.replace(".modules_to_save", "").replace(".default", "")
            state_dict[base_param_name].data.copy_(param / num_client)
    fronzen_model.load_state_dict(state_dict)

