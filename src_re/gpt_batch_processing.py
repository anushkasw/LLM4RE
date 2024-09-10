import json
import os
from pathlib import Path
import random
import argparse
import time

from gpt_api import Demo

import logging


def setup_logger(log_file_path_and_name, logger_name, level=logging.ERROR, formatter=None):
    log_path, _ = os.path.split(log_file_path_and_name)

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a file handler
    os.makedirs(log_path, exist_ok=True, mode=0o755)
    # file_handler = RotatingFileHandler(log_file_path_and_name)
    file_handler = logging.FileHandler(log_file_path_and_name)
    file_handler.setLevel(level)

    # Create a logging format
    if formatter:
        formatter = logging.Formatter(formatter)
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    return logger


def main(args):
    random.seed(args.seed)
    for data_seed in [100, 13, 42]:
        print(f'Evaluating Seed - {data_seed}')
        demo = Demo(
            api_key=args.api_key,
            engine=args.model,
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=True,
        )

        store_path = f'{args.output_dir}/RC/{args.model}/{args.task}/{args.demo}/seed-{data_seed}'
        os.makedirs(f'{store_path}/output', exist_ok=True)

        # Setup loggers
        logging_dir = f'{store_path}/logs/output'
        os.makedirs(logging_dir, exist_ok=True)
        config_done_logger = setup_logger(f'{logging_dir}/output_done_paths.log', 'config_done_logger', level=logging.INFO, formatter='%(message)s')
        config_redo_logger = setup_logger(f'{logging_dir}/output_redo_paths.log', 'config_redo_logger', level=logging.INFO, formatter='%(message)s')

        batch_files = list(Path(f'{store_path}/input').glob('*.jsonl'))
        print(f'Total batches = {len(batch_files)}')

        for file in batch_files:
            if not os.path.exists(f'{store_path}/output/{file.name}'):
                print(f'Processing: {file}')
                start_time = time.time()
                result = demo.process_batch(file)

                if result:
                    config_done_logger.info(f'{file},{(time.time()-start_time)/60}\n')
                    result_file_name = f'{store_path}/output/{file.name}'
                    with open(result_file_name, 'w') as f:
                        f.write(result + '\n')
                else:
                    config_redo_logger.info(file)
            else:
                print(f'Output already present for file - {file}')

        print('\n Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=False, help="OpenAI API")
    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--task', '-t', type=str, required=False, help="Dataset Name.")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False,
                        choices=['random', 'knn', 'zero', 'fast_votek'],
                        help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='OpenAI/gpt-4o-mini', required=False,
                        help="LLM")

    parser.add_argument("--output_dir", default='./batches', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")


    args = parser.parse_args()
    main(args)