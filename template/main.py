from tqdm import tqdm
import argparse
import math
import json
# import time

from gpt_api import Demo
from data_loader import DataProcessor
from prompt import create_prompt
# from pipelines_HF import HFModelPipelines
# from datetime import timedelta


def main(args):
    demo = Demo(
        api_key=args.api_key,
        engine=args.model,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
    )


    data_processor = DataProcessor(args)
    demo_dict = data_processor.get_train_examples()  # train data
    test_dict = data_processor.get_test_examples()

    test_examples = [item for sublist in test_dict.values() for item in sublist]


    test_res = []
    for input in tqdm(test_examples):
        prompt = create_prompt(args, input, demo_dict, data_processor)

        try:
            # Modified API call method and parameters for newer API
            result, probs = demo.get_multiple_sample(prompt)

            test_res.append({
                "id": input['id'],
                "label_true": input['relation'],
                "label_pred": result[0],
                "probs": math.exp(probs[0].content[0].logprob)
            })

        except Exception as e:
            print(e)
            if e._message == 'You exceeded your current quota, please check your plan and billing details.':
                continue

    with open(f'{args.out_path}/test.jsonl', 'w') as f:
        for res in test_res:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="OpenAI API")
    parser.add_argument('--data_dir', '-dir', type=str, required=True, help="The path of training / test data.")
    parser.add_argument('--out_path', '-out', type=str, default='./', required=False, help="Output Directory")
    parser.add_argument('--task', '-t', type=str, required=True, help="Dataset Name.")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False, help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='gpt-3.5-turbo', required=False, help="LLM")
    parser.add_argument('--prompt', type=str, required=False, default="instruct_schema",
                        choices=["text", "text_schema", "instruct", "instruct_schema"])
    parser.add_argument('--k', type=int, default=5, help="k-shot demonstrations")
    parser.add_argument('--data_seed', type=int, default=13, help="k-shot demonstrations")
    args = parser.parse_args()

    main(args)

