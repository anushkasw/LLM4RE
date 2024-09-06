from tqdm import tqdm
import argparse
import math
import random
import numpy as np
import json
import time
import os
import sys
import traceback

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
    for k in [5, 10, 20, 30]:
        print(f'\tEvaluating Shot - {k}')
        for data_seed in [100, 13, 42]:
            print(f'\tEvaluating Seed - {data_seed}')
            outpath = f'{args.out_path}/RC/{args.model}/{args.task}/{args.demo}/seed-{data_seed}'
            os.makedirs(outpath, exist_ok=True)

            data_processor = DataProcessor(args, data_seed)
            print(f'\tLoading training data')
            train_dict = data_processor.get_train_examples()  # train data
            print(f'\tLoading test data')
            test_dict = data_processor.get_test_examples()

            incomplete_flag = False
            if os.path.exists(f'{outpath}/{args.prompt}-{k}.jsonl'):
                with open(f'{outpath}/{args.prompt}-{k}.jsonl') as f:
                    batch = f.read().splitlines()
                test_completed = {json.loads(line)['id']: json.loads(line) for line in batch if line != ""}
                if len(test_completed) == len(test_dict):
                    print(f'\tResults already processed. Terminating')
                    sys.exit(1)
                if len(test_completed) != len(test_dict):
                    print(f'\tSome results already processed. Setting incomplete_flag to True')
                    incomplete_flag = True

            demo, model, tokenizer = None, None, None
            if args.pipe:
                demo = Demo_HF(
                    access_token=args.api_key,
                    model_name=args.model,
                    max_tokens=128,
                    cache_dir=args.cache_dir,
                )
            else:
                tokenizer, model = model_init(args.model, args.cache_dir)

            print(f'\tNumber of GPUs available: {torch.cuda.device_count()}')
            print(f'\tLoading Demo Mapping from: {args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl')
            if os.path.exists(f'{args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl'):
                with open(f'{args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl', 'r') as f:
                    demo_mapping = json.load(f)
            else:
                raise Exception(f'Cannot find {args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl')

            test_res = []
            for test_idx, input in tqdm(test_dict.items()):
                if incomplete_flag:
                    if input.id in test_completed:
                        continue
                demo_list = [train_dict[i] for i in demo_mapping[test_idx]]
                prompt = create_prompt(args, input, demo_list, data_processor)

                try:
                    if args.pipe:
                        result = demo.get_multiple_sample(prompt)
                    else:
                        result = model_inference(tokenizer, model, prompt, max_new_tokens=128, device='cuda')
                except Exception as e:
                    raise e

                test_res = {
                    "id": input.id,
                    "label_pred": result,
                }

                with open(f'{outpath}/{args.prompt}-{k}.jsonl', 'a') as f:
                    if f.tell() > 0:  # Check if file is not empty
                        f.write('\n')
                    json.dump(test_res, f)
            del data_processor, model, tokenizer, demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=False, help="Hugging Face API access token")
    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--task', '-t', type=str, required=False, help="Dataset Name.")
    parser.add_argument('--prompt', type=str, default='open', choices=['open', 'entrel', 'rel', 'ent'], help="Prompt Type")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False, help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=False, help="LLM")
    parser.add_argument("--pipe", action='store_true', help="if use huggingface pipeline")
    parser.add_argument("--reason", action='store_true', help="Add reasoning to examples")

    parser.add_argument('--data_dir', '-dir', type=str, required=False,
                        default="/blue/woodard/share/Relation-Extraction/Data")
    parser.add_argument('--prompt_dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/prompts")
    parser.add_argument('--out_path', '-out', type=str, default='./', required=False, help="Output Directory")
    parser.add_argument('--cache_dir', type=str, default="/blue/woodard/share/Relation-Extraction/LLM_for_RE/cache", help="LLM cache directory")

    parser.add_argument("--config_file", type=str, default=None,
                        help="path to config file", required=False)
    parser.add_argument('--redo', type=bool, default=False)
    args = parser.parse_args()

    if args.config_file:
        config_file = args.config_file
        with open(args.config_file, 'r') as f:
            args.__dict__ = json.load(f)
            setattr(args, 'config_file', config_file)

    try:
        main(args)
        if args.config_file or os.path.exists(
                f'{args.out_path}/redo_exps/RC/{args.task}/{args.model}/exp-{args.demo}_{args.prompt}.json'):
            os.remove(args.config_file)
    except Exception as e:
        print(f'[Error] {e}')
        print(traceback.format_exc())
        setattr(args, 'redo', True)
        redo_bin = f'{args.out_path}/redo_exps/RC/{args.task}/{args.model}'
        os.makedirs(redo_bin, exist_ok=True)
        with open(f'{redo_bin}/exp-{args.demo}_{args.prompt}.json', 'w') as f:
            json.dump(args.__dict__, f)

    print('\tDone.')
