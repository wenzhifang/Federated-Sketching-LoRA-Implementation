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


def generate_ranks(N, a=4, b=64, dist_type='uniform', seed=42):
    if seed is not None:
        np.random.seed(seed)
    if dist_type == 'uniform':
        samples = np.random.uniform(a, b, N)

    elif dist_type == 'normal':
        mu = (a + b) / 2
        sigma = (b - a) / 6  # ~99.7% within [a, b]
        samples = np.random.normal(mu, sigma, N)

    elif dist_type == 'heavy_tail_low':
        # Use inverse log-normal to bias toward low values
        mu = np.log((a + b) / 4)
        sigma = 1.0
        samples = 1 / np.random.lognormal(mu, sigma, N)
        samples = a + (samples - np.min(samples)) / (np.max(samples) - np.min(samples)) * (b - a)
        
    else:
        raise ValueError("Unsupported distribution type.")

    samples = np.clip(samples, a, b)
    return np.round(samples).astype(int).tolist()        
k_list = generate_ranks(args.clients, 4, args.lora_r, dist_type=args.rank_type)

model = models.build_model(base_model)
total = sum(p.numel() for p in model.parameters())

print(model.config.name_or_path)  # Shows the model identifier used to load it
print(model.config.model_type)    # S

peft_model = models.add_adapter(model, args.lora_r, args.lora_alpha)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Training {trainable} parameters ({100*trainable/total:.2f}% of original {total})")

client_dataloaders = build_datasets(args, alpha=0.1)

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
