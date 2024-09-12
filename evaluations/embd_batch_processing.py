import json
import os
from pathlib import Path
import argparse
import time

from gpt_api import Demo

import logging


def main(args):
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

    os.makedirs(f'{args.output_dir}/output', exist_ok=True)

    batch_files = list(Path(f'{args.output_dir}/input').glob('*.jsonl'))
    print(f'Total batches = {len(batch_files)}')

    for file in batch_files:
        if not os.path.exists(f'{args.output_dir}/output/{file.name}'):
            print(f'Processing: {file}')
            start_time = time.time()
            result = demo.process_batch(file)

            if result:
                result_file_name = f'{args.output_dir}/output/{file.name}'
                with open(result_file_name, 'w') as f:
                    f.write(result + '\n')
            else:
                continue
        else:
            print(f'Output already present for file - {file}')

        print('\n Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=False, help="OpenAI API")
    parser.add_argument("--output_dir", default='./batches', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")

    args = parser.parse_args()
    main(args)