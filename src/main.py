from arg import parse
import models
from LoRA_sketching_llama_het import fl_slora_train_llama_het
import random
import torch
from utils_data import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login("...")

from accelerate import Accelerator

args = parse()

base_seed = 42
random.seed(base_seed)
torch.manual_seed(base_seed)

accelerator = Accelerator(cpu=False)
'''
# For heterogeneous LoRA
random_numbers = [random.random() for _ in range(args.clients)]
sketch_list = [0.125, 0.25, 0.5]
k_list = []
for i in range(args.clients):
    if 2/3 <= random_numbers[i] <=1:
        k_list.append(int(args.lora_r*sketch_list[2]))
    elif 1/3 <= random_numbers[i] <2/3:
        k_list.append(int(args.lora_r*sketch_list[1]))
    else:
        k_list.append(int(args.lora_r*sketch_list[0]))
'''
model = models.build_model(base_model)
total = sum(p.numel() for p in model.parameters())

peft_model = models.add_adapter(model, args.lora_r, args.lora_alpha)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Training {trainable} parameters ({100*trainable/total:.2f}% of original {total})")

client_dataloaders = build_datasets(args, base_seed = base_seed)
# For Homogeneous LoRA
k_list = [int(x * args.sketching_ratio) for x in [args.lora_r] * args.clients]
print('k_list:', k_list)
print('rank:', args.lora_r)

fl_slora_train_llama_het(peft_model, client_dataloaders,
    server_opt=args.server_opt,
    server_lr=args.server_lr,
    client_lr=args.client_lr,
    eval_freq=args.eval_freq,
    r = args.lora_r,
    m_list = k_list,
    accelerator = accelerator,
    base_seed = base_seed
)
