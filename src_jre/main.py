from tqdm import tqdm
import argparse
import math
import random
import numpy as np
import json
import sys
import os
import traceback

import torch
import gc

gc.collect()
torch.cuda.empty_cache()

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
    if not args.demo == 'zero':
        k_list = [5, 10, 20]
    else:
        k_list = [0]
    for k in k_list:
        print(f'\tEvaluating Shot - {k}')
        for data_seed in [100, 13, 42]:
            print(f'\tEvaluating Seed - {data_seed}')
            if args.model != 'OpenAI/gpt-4o-mini':
                outpath = f'{args.out_path}/JRE/{args.model}/{args.task}/{args.demo}/seed-{data_seed}'
            else:
                outpath = f'{args.out_path}/JRE/{args.model}/{args.task}/{args.demo}/seed-{data_seed}/input'
            os.makedirs(outpath, exist_ok=True)

            data_processor = DataProcessor(args, data_seed)
            print(f'\tLoading training data')
            train_dict = data_processor.get_train_examples()  # train data
            print(f'\tLoading test data')
            test_dict = data_processor.get_test_examples()

            incomplete_flag = False
            if os.path.exists(f'{outpath}/{args.prompt}-{k}.jsonl'):
                if args.model != 'OpenAI/gpt-4o-mini':
                    with open(f'{outpath}/{args.prompt}-{k}.jsonl') as f:
                        batch = f.read().splitlines()
                    test_completed = {json.loads(line)['id']: json.loads(line) for line in batch if line != ""}
                else:
                    with open(f'{outpath}/{args.prompt}-{k}.jsonl') as f:
                        batch = f.read().splitlines()
                        test_completed = {json.loads(line)['custom_id']: json.loads(line) for line in batch if line != ""}
                if len(test_completed) == len(test_dict):
                    print(f'\tResults already processed. Terminating')
                    continue
                if len(test_completed) != len(test_dict):
                    print(f'\tSome results already processed. Setting incomplete_flag to True')
                    incomplete_flag = True

            if args.model != 'OpenAI/gpt-4o-mini':
                tokenizer, model = model_init(args.model, args.cache_dir)
            else:
                tokenizer, model = None, None

            print(f'\tNumber of GPUs available: {torch.cuda.device_count()}')

            if not args.demo == 'zero':
                print(f'\tLoading Demo Mapping from: {args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl')
                if os.path.exists(f'{args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl'):
                    with open(f'{args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl', 'r') as f:
                        demo_mapping = json.load(f)
                else:
                    raise FileNotFoundError(f'Cannot find {args.data_dir}/{args.task}/{args.demo}Demo/k-{k}.jsonl')

            for test_idx, input in tqdm(test_dict.items()):
                if incomplete_flag:
                    if input.id in test_completed:
                        continue

                demo_list = None
                if not args.demo == 'zero':
                    demo_list = [train_dict[i] for i in demo_mapping[test_idx]]
                prompt = create_prompt(args, input, demo_list, data_processor)

                if args.model != 'OpenAI/gpt-4o-mini':
                    try:
                        result = model_inference(tokenizer, model, prompt, max_new_tokens=300, device='cuda')
                    except Exception as e:
                        print(f'\n[Error] {e}')

                    test_res = {
                        "id": input.id,
                        "label_pred": result,
                    }

                    with open(f'{outpath}/{args.prompt}-{k}.jsonl', 'a') as f:
                        if f.tell() > 0:  # Check if file is not empty
                            f.write('\n')
                        json.dump(test_res, f)
                else:
                    try:
                        batch_dict = {"custom_id": input.id, "method": "POST", "url": "/v1/chat/completions",
                                      "body": {"model": "gpt-4o-mini",
                                               "messages": prompt, "temperature": 0,
                                               "max_tokens": 300, "logprobs": True}}
                        with open(f'{outpath}/{args.prompt}-{k}.jsonl', 'a') as f:
                            if f.tell() > 0:  # Check if file is not empty
                                f.write('\n')
                            json.dump(batch_dict, f)
                    except Exception as e:
                        print(f'\n[Error] {e}')

            del data_processor, model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--task', '-t', type=str, required=True, help="Dataset Name.")
    parser.add_argument('--prompt', type=str, default='open', choices=['open', 'entrel'], help="Prompt Type")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False,
                        choices=['random', 'knn', 'zero', 'fast_votek'],
                        help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=True,
                        help="LLM")
    parser.add_argument("--pipe", action='store_true', help="if use huggingface pipeline")
    parser.add_argument("--reason", action='store_true', help="Add reasoning to examples")

    parser.add_argument('--data_dir', '-dir', type=str, required=True,
                        default="/blue/woodard/share/Relation-Extraction/Data")
    parser.add_argument('--prompt_dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/prompts")
    parser.add_argument('--out_path', '-out', type=str, default='./', required=True, help="Output Directory")
    parser.add_argument('--cache_dir', type=str, default="/blue/woodard/share/Relation-Extraction/LLM_for_RE/cache",
                        help="LLM cache directory")
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
                f'{args.out_path}/redo_exps/JRE/{args.task}/{args.model}/exp-{args.demo}_{args.prompt}.json'):
            os.remove(f'{args.out_path}/redo_exps/JRE/{args.task}/{args.model}/exp-{args.demo}_{args.prompt}.json')
    except Exception as e:
        print(f'[Error] {e}')
        print(traceback.format_exc())
        setattr(args, 'redo', True)
        redo_bin = f'{args.out_path}/redo_exps/JRE/{args.task}/{args.model}'
        os.makedirs(redo_bin, exist_ok=True)
        with open(f'{redo_bin}/exp-{args.demo}_{args.prompt}.json', 'w') as f:
            json.dump(args.__dict__, f)
    print('\tDone.')
