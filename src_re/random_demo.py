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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_demonstrations(train_dict, k):
    train_list = [val for x, val in train_dict.items()]
    random.shuffle(train_list)
    demo_list = train_list[0:k]
    return [x.id for x in demo_list]

def get_demonstration_mappings(args):
    set_seed(args)
    outpath = f'{args.data_dir}/{args.task}/randomDemo'
    os.makedirs(outpath, exist_ok=True)

    data_processor = DataProcessor(args)
    # for seed in [13, 42, 100]:
    for k in [1, 5, 10, 20, 30]:
        train_dict = data_processor.get_train_examples()  # train data
        # train_dict = dict(random.sample(train_dict.items(), 200))
        test_dict = data_processor.get_test_examples()
        # test_dict = dict(random.sample(test_dict.items(), 100))

        demo_mappings = {}

        for test_idx, input in tqdm(test_dict.items()):
            demo_list = get_demonstrations(train_dict, k)
            demo_mappings[input.id] = demo_list

        with open(f'{outpath}/k-{k}.jsonl', 'w+') as f:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(demo_mappings, f)

        del test_dict, train_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Demo Retreiver')
    parser.add_argument("--use_cuda", action='store_true',
                        help="if GPUs available")
    # Required
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--reason", action='store_true', help="Add reasoning to examples")

    # Training Parameters
    parser.add_argument('--na_idx', type=int, default=None)
    parser.add_argument("--no_na", action='store_true',
                        help="if na samples should not be included")

    args = parser.parse_args()
    get_demonstration_mappings(args)
    print('\tDone.')