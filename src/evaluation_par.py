import os
import copy
import json
import torch
import models

from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer
from utils_data import *
accelerator = Accelerator(cpu=False)  # Force GPU usage

from transformers import LlamaForCausalLM


def evaluate_mini_batch(model,
            instructions,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        inputs = generate_tokenizers_eval(instructions)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        #### for safe 
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)
    
        generation_config = GenerationConfig(
            #do_sample=True, #### delete
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                pad_token_id=tokenizer.pad_token_id,  # Explicitly set the pad_token_id
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        # print(outputs)
        return outputs

def evaluate(model, args, rnd):
    if accelerator.is_main_process:
        # Log the global accuracy
        save_path_log = f'./training_log_set/{args.algorithm}/{rnd}/{args.dataset}'
        os.makedirs(save_path_log, exist_ok=True)
        log_eval = f'./training_log_set/{args.algorithm}/{rnd}/{args.dataset}/eval_SLoRA_dataset_{args.dataset}.txt'
        with open(log_eval, 'w') as log_file:
            log_file.write('Starting: {} \n'.format(args.dataset))
    # Synchronize processes to ensure the directory is created before proceeding
    accelerator.wait_for_everyone()
    
    model.eval()
    valloader = build_datasets_eval(args)
    valloader = accelerator.prepare(valloader)
    local_correct = 0
    local_total = 0
    for idx, batch in enumerate(valloader):
        local_total += len(batch['instruction'])
        instructions = batch['instruction']
        outputs = evaluate_mini_batch(model, instructions)

        for instruction, label, output in zip(batch['instruction'], batch['answer'], outputs):
            predict = extract_answer(args, output)
            if label == predict:
                local_correct += 1
        
    
    local_correct_t = torch.tensor(local_correct, device=accelerator.device, dtype=torch.float)
    local_total_t   = torch.tensor(local_total, device=accelerator.device, dtype=torch.float)

    global_correct  = accelerator.reduce(local_correct_t, reduction="sum")
    global_total    = accelerator.reduce(local_total_t, reduction="sum")
    if accelerator.is_main_process:
        mean_accuracy = global_correct / global_total
        with open(log_eval, 'a') as log_file:
            log_file.write('eval accuracy {:.3f}\n'.format(mean_accuracy.item()))
    
    accelerator.free_memory()         

if __name__ == '__main__':
    args = parse()
    base_seed = 42
    torch.manual_seed(base_seed)
    pbar = tqdm(range(14, -1, -1), desc=f'{args.dataset}')
    for rnd in pbar:
        save_model_path = '...'
        model = models.build_model(args.base_model)
        peft_model = PeftModel.from_pretrained(model, save_model_path)
        evaluate(peft_model, args, rnd) 
        
