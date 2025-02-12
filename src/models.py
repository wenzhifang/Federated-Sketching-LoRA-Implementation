import torch
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel
from transformers import BitsAndBytesConfig
from utils_data import *


def build_model(base_model):
    config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            use_auth_token = True, 
            quantization_config=config,
            cache_dir="..." # model saving path
        )
    model = prepare_model_for_kbit_training(model)
    return model

def add_adapter(model, r, alpha, target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]):
    config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True
        )
    model_adapter = PeftModel(model, config)
    return model_adapter






