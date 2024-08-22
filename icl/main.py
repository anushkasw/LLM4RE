import argparse
import random
import os
import numpy as np
import json

import torch
import torch.nn as nn

from tqdm import tqdm

import gc
gc.collect()
torch.cuda.empty_cache()

from data_loader import DataProcessor
from knn import get_demonstrations, generate_ft_example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main(args):
    set_seed(args)

    data_processor = DataProcessor(args)
    demo_dict = data_processor.get_train_examples()  # train data
    test_dict = data_processor.get_test_examples()
    test_examples = [item for sublist in test_dict.values() for item in sublist]

    if args.demo_ret=='knn':
        ## Create training prompts and demonstration retrieval models
        ft_dict, gpu_index_flat, train_dict, train_sentences, knn_model = get_demonstrations(args, demo_dict, reltoid)

        for input in tqdm(test_examples):
            demo_list = generate_ft_example(tmp_dict, ft_dict, reltoid, idtoprompt, demo, args)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MentalLlama')
    parser.add_argument("--use_cuda", action='store_true',
                        help="if GPUs available")
    # Required
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--demo_ret', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    # Not in config required
    parser.add_argument('--data_save_path', type=str, default='./')

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=500)

    args = parser.parse_args()
    main(args)