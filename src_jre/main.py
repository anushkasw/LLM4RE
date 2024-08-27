import os.path

from tqdm import tqdm
import argparse
import math
import json
import random

from demo_HF import Demo_HF
from data_loader import DataProcessor
from prompt import create_prompt

def main(args):
    demo = Demo_HF(
        cache_dir=args.cache_dir,
        access_token=args.api_key,
        model_name=args.model,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
    )

    data_processor = DataProcessor(args)
    print(f'\tLoading training data')
    train_dict = data_processor.get_train_examples()  # train data
    print(f'\tLoading test data')
    test_dict = data_processor.get_test_examples()

    print(f'\tLoading Demo Mapping')
    if os.path.exists(f'{args.data_dir}/{args.task}/{args.demo}/k-{args.k}.jsonl'):
        with open(f'{args.data_dir}/{args.task}/{args.demo}/k-{args.k}.jsonl', 'r') as f:
            demo_mapping = json.load(f)
    else:
        raise FileNotFoundError(f'Cannot find {args.data_dir}/{args.task}/{args.demo}/k-{args.k}.jsonl')

    test_res = []
    for test_idx, input in tqdm(test_dict.items()):
        demo_list = [train_dict[i] for i in demo_mapping[test_idx]]
        prompt = create_prompt(args, input, demo_list, data_processor)

        try:
            result, logprobs = demo.get_multiple_sample(prompt)
            test_res.append({
                "id": input['id'],
                "label_true": input['relation'],
                "label_pred": result[0],
                "probs": math.exp(logprobs[0][0].max().item()) if logprobs else None
            })

        except Exception as e:
            print(e)
            if hasattr(e, '_message') and e._message == 'You exceeded your current quota, please check your plan and billing details.':
                continue

    with open(f'{args.out_path}/test.jsonl', 'w') as f:
        for res in test_res:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--data_dir', '-dir', type=str, required=True, default="/blue/woodard/share/Relation-Extraction/Data")
    parser.add_argument('--out_path', '-out', type=str, default='./', required=True, help="Output Directory")
    parser.add_argument('--task', '-t', type=str, required=True, help="Dataset Name.")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False, help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=True, help="LLM")

    parser.add_argument("--entity_info", action='store_true', help="entity type information to be added to the prompt")
    parser.add_argument("--rel_info", action='store_true', help="relation type information to be added to the prompt")
    parser.add_argument('--k', type=int, default=1, help="k-shot demonstrations")
    parser.add_argument('--data_seed', type=int, default=13, help="k-shot demonstrations")
    parser.add_argument('--cache_dir', type=str, default="/blue/woodard/share/Relation-Extraction/LLM_for_RE/cache", help="LLM cache directory")
    args = parser.parse_args()

    try:
        main(args)
    except FileNotFoundError as e:
        print(e)

    print('\tDone.')
