import json
import argparse
# from openai_emb import embedding_retriever
from collections import defaultdict
# from tqdm import tqdm
# import threading
import os
from pathlib import Path
from utils import sanity_check

def get_elements(triple_list, element_set):
    for triple in triple_list:
        if type(triple[0]) == list:
            for triple_ in triple:
                for element in triple_:
                    element_set.add(element)
        else:
            for element in triple:
                element_set.add(element.strip())
def main(args):
    element_set = set()

    for data in ['NYT10', 'tacred', 'crossRE', 'FewRel']:
        with open(f'/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE/{data}/test.jsonl', "r") as f:
            for line in f.read().splitlines():
                sample = json.loads(line)
                triple_list = []
                for triple in sample['relationMentions']:
                    relation = triple['label'].replace('_', ' ').replace('/', ' ').replace("-", " ").strip()
                    triple_list.append([triple['em1Text'], relation, triple['em2Text']])
                get_elements(triple_list, element_set)

        for model in ["openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407",
                      "google/gemma-2-9b-it"]:
            files = list(
                Path(f'{args.base_path}/processed_results/JRE/{data}/{model}'
                     ).rglob('*.jsonl'))

            for file in files:
                prompt = file.parts[-1].split('-')[0]
                dataset = file.parts[-6]

                check = sanity_check(args.exp, dataset, prompt)
                if check:
                    with open(file, "r") as f:
                        for line in f.read().splitlines():
                            sample = json.loads(line)
                            get_elements(sample['pred_label'], element_set)

    batches = []
    for element in element_set:
        batches.append({"custom_id": element, "method": "POST", "url": "/v1/embeddings",
     "body": {"model": "text-embedding-3-large", "input": element}})

    batch_size = 10000
    outpath = f'{args.base_path}/embd_batches'
    os.makedirs(outpath, exist_ok=True)
    for i in range(0, len(batches), batch_size):
        batch = batches[i:i + batch_size]
        with open(f'{outpath}/input_{i+1}-{i + batch_size}.jsonl', 'w') as file:
            for item in batch:
                json.dump(item, file)
                file.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="jre")
    parser.add_argument('--ts', type=bool, default=False)
    # parser.add_argument('--model_name', '-m', type=str, required=False, help="Model Name.", default="mistral")
    #
    parser.add_argument('--base_path', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25")
    # parser.add_argument("--output_dir", default='./output', type=str, required=False,
    #                     help="The output directory where the lda model")

    args = parser.parse_args()
    main(args)