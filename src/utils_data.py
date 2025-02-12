from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import re
import random
import os
import torch
from arg import parse
args = parse()
base_model = args.base_model

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(base_model, token = True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
def tokenize(prompt, cutoff_len):
    
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    
    if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    
    return result

def generate_and_tokenize_prompt(data_point, train_on_inputs=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, cutoff_len=256)
    if not train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
                                                -100
                                            ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                user_prompt_len:
                                                                ]  # could be sped up, probably
    return tokenized_full_prompt

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    
def collate_fn(batch):
    # Extract input_ids, attention_mask, and labels from each item in the batch
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask']) for x in batch]
    labels = [torch.tensor(x['labels']) for x in batch]

    # Pad sequences to the same length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 for ignored tokens

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

def collate_fn_left(batch):
    # Extract input_ids, attention_mask, and labels from each item in the batch
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask']) for x in batch]
    labels = [torch.tensor(x['labels']) for x in batch]

    # Flip sequences for left padding
    input_ids_flipped = [seq.flip(0) for seq in input_ids]
    attention_mask_flipped = [seq.flip(0) for seq in attention_mask]
    labels_flipped = [seq.flip(0) for seq in labels]

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_flipped, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    ).flip(1)  # flip back after padding
    
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_flipped, 
        batch_first=True, 
        padding_value=0
    ).flip(1)  # flip back after padding
    
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_flipped, 
        batch_first=True, 
        padding_value=-100
    ).flip(1)  # flip back after padding

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

def build_datasets(args, base_seed):
    file_path = f'./dataset/{args.dataset}/train.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    dataset = load_dataset("json", data_files=file_path)["train"]
    trainset = dataset.map(generate_and_tokenize_prompt)

    # Convert dataset to a list for shuffling
    trainset_list = list(trainset)
    random.seed(base_seed)
    # Perform deterministic shuffling (seed already set in main.py)
    random.shuffle(trainset_list)  # Use Python's random.shuffle

    # Convert the shuffled list back to a dataset
    shuffled_trainset = trainset.select(range(len(trainset_list)))

    # Shard the dataset into client-specific subsets
    clients = [
        DataLoader(
            shuffled_trainset.shard(num_shards=args.clients, index=i),
            batch_size=args.client_batch,
            shuffle=False,  # No need to shuffle again here
            num_workers=4,
            collate_fn=collate_fn_left,
            pin_memory=True,
        )
        for i in range(args.clients)
    ]
    return clients


## below for evaluation

def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


def generate_prompt_eval(instruction):
    
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

            ### Instruction:
            {instruction}

            ### Response:

# def generate_tokenizers_eval(instructions):
#     prompts = [generate_prompt_eval(instruction) for instruction in instructions]
#     return tokenizer(prompts, return_tensors="pt", padding=True)
def generate_tokenizers_eval(instructions):
    prompts = [generate_prompt_eval(instruction) for instruction in instructions]
    results = tokenizer(prompts, return_tensors="pt", padding=True)
    return results
    ''' 
    results = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = results["input_ids"]
    attention_mask = results["attention_mask"]

    # Check and append eos_token_id to each input sequence if it's not already there
    eos_token_id = tokenizer.eos_token_id
    # Create lists to hold updated input_ids and attention_mask
    new_input_ids = []
    new_attention_mask = []

    for i in range(input_ids.shape[0]):
        # Append eos_token_id if not present
        if input_ids[i, -1] != eos_token_id:
            padded_input = F.pad(input_ids[i], (0, 1), value=eos_token_id)
            padded_mask = F.pad(attention_mask[i], (0, 1), value=1)
        else:
            padded_input = input_ids[i]
            padded_mask = attention_mask[i]

        new_input_ids.append(padded_input)
        new_attention_mask.append(padded_mask)

    # Stack back into tensors
    results["input_ids"] = torch.stack(new_input_ids, dim=0)
    results["attention_mask"] = torch.stack(new_attention_mask, dim=0)

    return results
    '''

def build_datasets_eval(args):
    batch_size = args.test_batch
    file_path = f'./dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    dataset = load_dataset("json", data_files=file_path)
    valloader = DataLoader(dataset["train"], batch_size = batch_size, num_workers=0, pin_memory=True, shuffle=False)
    return valloader
