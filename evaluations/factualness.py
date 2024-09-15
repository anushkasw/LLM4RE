import argparse
import json
import os
from pathlib import Path

from utils import sanity_check
from data_loader import get_RC_data, get_JRE_data


def create_batches(tmp_dict, data_dict, prompt_path):
    with open(prompt_path, 'r') as f:
        fact_checker_prompt = f.read().strip()

    prompt_batchs = []
    for idx, dict_ in tmp_dict.items():
        source_text = data_dict[idx]['text']
        triples = dict_['pred_label']

        prompt = fact_checker_prompt.replace('$TEXT$', source_text).replace('$TRIPLE$', json.dumps(triples))
        messages = [{"role": "user", "content": prompt}]
        batch_dict = {"custom_id": idx, "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-3.5-turbo-0125",
                               "messages": messages, "temperature": 0.3,
                               "max_tokens": 10}}
        prompt_batchs.append(batch_dict)
    return prompt_batchs


def main(args):
    out_path = f'{args.base_path}/factual_batches'
    os.makedirs(out_path, exist_ok=True)

    for data in ['NYT10', 'tacred', 'crossRE', 'FewRel']:
        print(data)
        if args.exp == 'JRE':
            data_dict, rel2id = get_JRE_data(data, args.data_dir)
        else:
            data_dict, rel2id = get_RC_data(data, args.data_dir)

        for model in ["openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "mistralai/Mistral-Nemo-Instruct-2407",
                      "google/gemma-2-9b-it", "OpenAI/gpt-4o-mini"]:
            print(model)
            files = list(
                Path(f'{args.base_path}/processed_results/{args.exp}/{data}/{model}'
                     ).rglob('*.jsonl'))

            for file in files:
                # print(file)
                prompt = file.parts[-1].split('-')[0]
                k = file.parts[-1].split('-')[-1].split('.')[0]
                seed = file.parts[-2].split('-')[-1]
                demo = file.parts[-3]
                llm = file.parts[-4]
                llm_fam = file.parts[-5]
                dataset = file.parts[-6]

                check = sanity_check(args.exp, dataset, prompt)
                if check:
                    tmp_dict = {}
                    with open(file, "r") as f:
                        for line in f.read().splitlines():
                            sample = json.loads(line)
                            tmp_dict[sample['id']] = sample

                    batches = create_batches(tmp_dict, data_dict,
                                             '/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/prompts/fact_checker.txt')
                    os.makedirs(f'{out_path}/JRE/{model}/{data}/{demo}/seed-{seed}/input', exist_ok=True)
                    with open(f'{out_path}/JRE/{model}/{data}/{demo}/seed-{seed}/input/{prompt}-{k}.jsonl', 'w') as f:
                        for batch in batches:
                            if f.tell() > 0:  # Check if file is not empty
                                f.write('\n')
                            json.dump(batch, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="JRE")
    #
    parser.add_argument('--base_path', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25")
    parser.add_argument("--data_dir", default='/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE', type=str,
                        required=False,
                        help="raw data dir")

    args = parser.parse_args()
    main(args)
    print('Done.')
