import argparse
import random
import os
import numpy as np
import json

import torch

from tqdm import tqdm

import gc
gc.collect()
torch.cuda.empty_cache()

from data_loader import DataProcessor
from knn import get_demonstrations, find_knn_example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main(args):
    set_seed(args)

    outpath = f'{args.data_save_path}/{args.task}/{args.demo_ret}'
    os.makedirs(outpath, exist_ok=True)

    data_processor = DataProcessor(args)
    for k in [1, 5, 10, 20, 30]:
        for seed in [13, 42, 100]:
            train_path = f'{args.data_dir}/{args.task}/train.json'
            test_path = f'{args.data_dir}/{args.task}/k-shot/seed-{seed}/test.json'

            demo_dict = data_processor.get_train_examples(train_path)  # train data
            test_dict = data_processor.get_test_examples(test_path)
            test_examples = [item for sublist in test_dict.values() for item in sublist]

            demo_mappings = {}
            if args.demo_ret=='knn':
                ## Create training prompts and demonstration retrieval models
                train_dict, knn_model = get_demonstrations(args, demo_dict)

                for input in tqdm(test_examples):
                    demo_list = find_knn_example(knn_model, input, train_dict, k)
                    demo_mappings[input['id']] = [x['id'] for x in demo_list]

                with open(f'{outpath}/{seed}-{k}.jsonl', 'a') as f:
                    if f.tell() > 0:  # Check if file is not empty
                        f.write('\n')
                    json.dump(demo_mappings, f)

                del demo_dict, test_dict, knn_model, train_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MentalLlama')
    parser.add_argument("--use_cuda", action='store_true',
                        help="if GPUs available")
    # Required
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--demo_ret', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    # Not in config required
    parser.add_argument('--data_save_path', type=str, default='./demo_dir')

    # Training Parameters
    parser.add_argument('--na_idx', type=int, default=None)
    parser.add_argument("--no_na", action='store_true',
                        help="if na samples should not be included")

    args = parser.parse_args()
    main(args)
    print('\tDone.')