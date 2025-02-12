import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    default="0,1",     type=str)
    parser.add_argument('--base-model',    default='meta-llama/Llama-3.2-3B-Instruct',  type=str) # 'meta-llama/Llama-2-7b-hf'; 'meta-llama/Llama-3.2-1B' meta-llama/Llama-3.2-3B-Instruct google/gemma-2-2b-it
    parser.add_argument('--dataset',    default='boolq',  type=str)
    parser.add_argument('--clients',    default=10,       type=int) #50, 20
    parser.add_argument('--server-opt',       default='adam',  type=str) #SGD
    parser.add_argument('--server-lr',        default=3e-4,    type=float) #3e-4
    parser.add_argument('--client-lr',        default=3e-4,    type=float) #5e-4
    parser.add_argument('--client-batch',     default=4,      type=int)
    parser.add_argument('--test-batch',     default=16,      type=int)
    parser.add_argument('--local_iter_per_round',     default=20,      type=int)
    parser.add_argument('--num_epochs',    default=100,       type=int) #20
    parser.add_argument('--lora_r',     default=16,      type=int)
    parser.add_argument('--lora-alpha', default=1, type=int)
    parser.add_argument('--sketching_ratio',     default=1,      type=float)
    parser.add_argument('--eval-freq',  default=5,         type=int) #10
    parser.add_argument('--algorithm',       default='slora',  type=str)
    return parser.parse_args()
