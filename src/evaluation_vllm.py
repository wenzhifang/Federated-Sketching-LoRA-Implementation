import os
import copy
import json
import torch
import models
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer
from utils_data import *

from transformers import LlamaForCausalLM
from vllm import LLM, SamplingParams
import shutil

def evaluate_mini_batch(llm, instructions, 
                        temperature=0.1, 
                        top_p=0.75, 
                        max_new_tokens=32, 
                        **kwargs):
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        **kwargs
    )
    with torch.no_grad():
        prompts = [generate_prompt_eval(instruction) for instruction in instructions]
        outputs = llm.generate(prompts, sampling_params)
    responses = [o.outputs[0].text.strip() for o in outputs]
    paired_data = []
    for instr, resp in zip(instructions, responses):
        paired_data.append({
            "instruction": instr,
            "response": resp
        })

    return responses

def evaluate(llm, args, rnd):
    save_path_log = ""
    os.makedirs(save_path_log, exist_ok=True)
    log_eval = f'{save_path_log}/eval.txt'

    if rnd==0:
        with open(log_eval, 'w') as log_file:
            log_file.write(f'Starting evaluation log.\n')
    
    valloader = build_datasets_eval(args)
    local_correct = 0
    local_total = 0
    for idx, batch in enumerate(valloader):
        local_total += len(batch['instruction'])
        instructions = batch['instruction']
        outputs = evaluate_mini_batch(llm, instructions)
        
        for instruction, label, output in zip(batch['instruction'], batch['answer'], outputs):
            predict = extract_answer(args, output)
            if label == predict:
                local_correct += 1
        
    mean_accuracy = local_correct / local_total
    with open(log_eval, 'a') as log_file:
        log_file.write(f'{rnd}  eval accuracy {mean_accuracy * 100:.2f}\n')
        
    
    
if __name__ == '__main__':
    args = parse()
    base_seed = 42
    torch.manual_seed(base_seed)
    pbar = tqdm(range(14, 15), desc=f'{args.algorithm}')
    
    
    for rnd in pbar:
        # Load full-precision base model (no quantization here!)
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        base_model_dir = ""
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=True
        )
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
    
    
        lora_path = ""
        iteration_model_path = os.path.join(base_model_dir, f"slora_{args.rank_type}", f"iteration_{rnd}")
        tokenizer.save_pretrained(iteration_model_path)

        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        model_with_lora = peft_model.merge_and_unload()
        
        model_with_lora.eval()
        
        model_with_lora.save_pretrained(iteration_model_path)
        
        llm = LLM(model=iteration_model_path, dtype="bfloat16", tensor_parallel_size=1)
        
        for dataset in ["boolq", "winogrande", "openbookqa", "ARC-Easy", "ARC-Challenge", "social_i_qa", "piqa", "hellaswag"]:
            args.dataset = dataset
            evaluate(llm, args, rnd) 
            
        shutil.rmtree(iteration_model_path)
        del peft_model, model_with_lora, llm
        torch.cuda.empty_cache()
        
