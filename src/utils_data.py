from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Subset
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

from huggingface_hub import login
login()

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
    # sorry about the formatting disaster gotta move fast
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
            """  # noqa: E501
            
def generate_tokenizers_eval(instructions):
    prompts = [generate_prompt_eval(instruction) for instruction in instructions]
    results = tokenizer(prompts, return_tensors="pt", padding=True)
    return results

def build_datasets_eval(args):
    batch_size = args.test_batch
    file_path = f'./dataset/{args.dataset}/test.json'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    
    dataset = load_dataset("json", data_files=file_path)
    data = dataset["train"]
    
    # Ensure dataset size is no larger than 3000
    max_samples = 3000
    if len(data) > max_samples:
        data = Subset(data, range(max_samples))
    
    valloader = DataLoader(data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return valloader



def build_datasets(args, alpha=0.5):
    num_clients = args.clients
    dataset_list = ["ARC-Challenge", "ARC-Easy", "boolq", "hellaswag", "openbookqa", "piqa", "social_i_qa", "winogrande"]
    base_seed = 42
    np.random.seed(base_seed)    
    datasets = {}
    for dataset in dataset_list:
        file_path = f'./dataset/{dataset}/train.json'
        raw_dataset = load_dataset("json", data_files=file_path)["train"]
        raw_dataset = list(raw_dataset)

        shuffled_indices = np.random.permutation(len(raw_dataset))
        datasets[dataset] = [raw_dataset[i] for i in shuffled_indices]

    
    # Calculate target size per client (average across all data)
    max_size = sum([len(dataset) for key, dataset in datasets.items()])//num_clients   
    #print(f"dataset size: {[len(dataset) for key, dataset in datasets.items()]}")
    
    largest_dataset_name = max(datasets.keys(), key=lambda x: len(datasets[x]))
    largest_dataset = datasets.pop(largest_dataset_name)
    
    clients = {i: [] for i in range(num_clients)}
    client_task_counts = {i: {dataset: 0 for dataset in datasets.keys()} for i in range(num_clients)}
    client_task_counts = {i: dict(client_task_counts[i], **{largest_dataset_name: 0}) for i in range(num_clients)}
    
    task_totals = {dataset: 0 for dataset in datasets.keys()}
    task_totals[largest_dataset_name] = 0
    
    for dataset_name, dataset in datasets.items():
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        ds_size = len(dataset)
        client_sample_sizes = [int(prop * ds_size) for prop in props]
        client_sample_sizes[-1] = ds_size - sum(client_sample_sizes[:-1])
        
        idx = 0
        for client_id, sample_size in enumerate(client_sample_sizes):
            if sample_size > 0:
                clients[client_id].extend(dataset[idx:idx+sample_size])
                client_task_counts[client_id][dataset_name] = sample_size
                task_totals[dataset_name] += sample_size
                idx += sample_size

    #print("Initial data distribution:", [len(clients[x]) for x in range(num_clients)])
    
    # Sort clients by size to fill smallest ones first
    sort_id = sorted(range(num_clients), key=lambda x: len(clients[x]))
    
    # Fill clients with largest dataset to reach target size
    largest_idx = 0
    for client_id in sort_id:
        size_diff = max_size - len(clients[client_id])
        if size_diff > 0 and largest_idx < len(largest_dataset):
            # Don't allocate more than what's available in largest dataset
            size_diff = min(size_diff, len(largest_dataset) - largest_idx)
            clients[client_id].extend(largest_dataset[largest_idx:largest_idx+size_diff])
            client_task_counts[client_id][largest_dataset_name] = size_diff
            task_totals[largest_dataset_name] += size_diff
            largest_idx += size_diff
    '''
    print("Final data distribution:", [len(clients[x]) for x in range(num_clients)])
    
    # Print distribution table
    sorted_datasets = sorted(list(datasets.keys()) + [largest_dataset_name])
    header = "Client | " + " | ".join(sorted_datasets) + " | Total"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    
    client_totals = []
    for client_id in range(num_clients):
        client_total = sum(client_task_counts[client_id].values())
        client_totals.append(client_total)
        row_values = [str(client_task_counts[client_id][ds]) for ds in sorted_datasets]
        print(f"{client_id:6d} | {' | '.join(row_values):s} | {client_total}")
    
    print(separator)
    task_total_values = [str(task_totals[ds]) for ds in sorted_datasets]
    print(f"Total  | {' | '.join(task_total_values)} | {sum(client_totals)}")
    print(separator)
    '''
    
    # Convert to Dataset objects and apply tokenization
    client_datasets = {i: Dataset.from_list(data) for i, data in clients.items()}
    client_datasets = [dataset.map(generate_and_tokenize_prompt) for i, dataset in client_datasets.items()]

    clients=[DataLoader(client_datasets[i],
            batch_size=args.client_batch,
            num_workers=4,
            collate_fn=collate_fn_left,
            pin_memory=True)
        for i in range(num_clients)
    ]

    return clients
