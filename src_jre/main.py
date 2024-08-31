from tqdm import tqdm
import argparse
import math
import random
import numpy as np
import json
import time
import os

import torch
import gc

gc.collect()
torch.cuda.empty_cache()

from demo_HF import Demo_HF
from data_loader import DataProcessor
from prompt import create_prompt
from no_pipe import model_init, model_inference

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def main(args):
    # if args.pipe:
    #     demo = Demo_HF(
    #         access_token=args.api_key,
    #         model_name=args.model,
    #         max_tokens=128,
    #         cache_dir=args.cache_dir,
    #     )
    # else:
    #     tokenizer, model = model_init(args.model, args.cache_dir)

    print(f'\tNumber of GPUs available: {torch.cuda.device_count()}')

    outpath = f'{args.out_path}/JRE/{args.model}/{args.task}/{args.demo}'
    os.makedirs(outpath, exist_ok=True)

    data_processor = DataProcessor(args)
    print(f'\tLoading training data')
    train_dict = data_processor.get_train_examples()  # train data
    print(f'\tLoading test data')
    test_dict = data_processor.get_test_examples()

    print(f'\tLoading Demo Mapping')
    if os.path.exists(f'{args.data_dir}/{args.task}/{args.demo}Demo/k-{args.k}.jsonl'):
        with open(f'{args.data_dir}/{args.task}/{args.demo}Demo/k-{args.k}.jsonl', 'r') as f:
            demo_mapping = json.load(f)
    else:
        raise FileNotFoundError(f'Cannot find {args.data_dir}/{args.task}/{args.demo}Demo/k-{args.k}.jsonl')

    test_res = []
    for test_idx, input in tqdm(test_dict.items()):
        demo_list = [train_dict[i] for i in demo_mapping[test_idx]]
        prompt = create_prompt(args, input, demo_list, data_processor)

        try:
            if args.pipe:
                result = demo.get_multiple_sample(prompt)
            else:
                result = model_inference(tokenizer, model, prompt, device='cuda')
        except Exception as e:
            print(f'\n[Error] {e}')

        test_res = {
            "id": input.id,
            "label_pred": result,
        }

        with open(f'{outpath}/{args.prompt}-{args.k}.jsonl', 'a') as f:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(test_res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--task', '-t', type=str, required=True, help="Dataset Name.")
    parser.add_argument('--k', type=int, default=1, help="k-shot demonstrations")
    parser.add_argument('--prompt', type=str, default='open', choices=['open', 'entrel'], help="Prompt Type")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False, help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=True, help="LLM")
    parser.add_argument("--pipe", action='store_true', help="if use huggingface pipeline")
    parser.add_argument("--reason", action='store_true', help="Add reasoning to examples")

    parser.add_argument('--data_dir', '-dir', type=str, required=True,
                        default="/blue/woodard/share/Relation-Extraction/Data")
    parser.add_argument('--prompt_dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/prompts")
    parser.add_argument('--out_path', '-out', type=str, default='./', required=True, help="Output Directory")
    parser.add_argument('--data_seed', type=int, default=13, help="k-shot demonstrations")
    parser.add_argument('--cache_dir', type=str, default="/blue/woodard/share/Relation-Extraction/LLM_for_RE/cache", help="LLM cache directory")
    args = parser.parse_args()

    main(args)

    print('\tDone.')
