import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    default="0,1",     type=str)
    parser.add_argument('--base-model',    default='meta-llama/Llama-3.2-3B-Instruct',  type=str) # 'meta-llama/Llama-2-7b-hf'; 'meta-llama/Llama-3.2-1B' meta-llama/Llama-3.2-3B-Instruct microsoft/Phi-3.5-mini-instruct google/gemma-2-2b-it
    parser.add_argument('--dataset',    default='boolq',  type=str) # used for evaluation
    parser.add_argument('--rank_type',    default='normal',  type=str)
    parser.add_argument('--clients',    default=50,       type=int) #
    parser.add_argument('--server_batch',    default=10,       type=int) #
    parser.add_argument('--server-opt',       default='adam',  type=str)
    parser.add_argument('--server-lr',        default=3e-4,    type=float) #3e-4
    parser.add_argument('--client-lr',        default=3e-4,    type=float) #5e-4
    parser.add_argument('--client-batch',     default=4,      type=int)
    parser.add_argument('--test-batch',     default=16,      type=int)
    parser.add_argument('--local_iter_per_round',     default=20,      type=int)
    parser.add_argument('--num_comm_rounds',     default=50,      type=int)
    parser.add_argument('--num_epochs',    default=15,       type=int) #20
    parser.add_argument('--lora_r',     default=64,      type=int)
    parser.add_argument('--lora-alpha', default=1, type=int)
    parser.add_argument('--sketching_ratio',     default=1,      type=float)
    parser.add_argument('--eval_freq',  default=1,         type=int) #10
    parser.add_argument('--algorithm',       default='slora',  type=str)
    parser.add_argument('--compression_ratio',     default=1,      type=float)
    parser.add_argument('--participation',     default=3,      type=int)
    parser.add_argument('--llama7_data',     default="dolly",      type=str)

    return parser.parse_args()
