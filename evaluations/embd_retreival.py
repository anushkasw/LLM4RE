import json
import pickle
import argparse
# from openai_emb import embedding_retriever
from collections import defaultdict
# from tqdm import tqdm
# import threading
import os
from pathlib import Path
from utils import sanity_check

# if not os.path.exists('/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict.pkl'):
#     embd_dict = None
# else:
#     with open('/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict.pkl', 'rb') as handle:
#         embd_dict = pickle.load(handle)

def get_embd_list():
    embd_list = {}
    files = list(
        Path(f'/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/embd_batches/input'
             ).rglob('*.jsonl'))

    for file in files:
        with open(file) as f:
            res_dict = f.read().splitlines()
        res_dict = [json.loads(line) for line in res_dict if line != '']

        for res in res_dict:
            embd_list[res['custom_id']] = res
    return embd_list


def main(args):
    embd_list = get_embd_list()
    element_set = set()

    def get_elements(triple_list, embd_list):
        for triple in triple_list:
            if type(triple[0]) == list:
                for triple_ in triple:
                    for element in triple_:
                        if embd_list:
                            if element not in embd_list:
                                element_set.add(element.strip())
                        else:
                            element_set.add(element.strip())
            else:
                for element in triple:
                    if embd_list:
                        if element not in embd_list:
                            element_set.add(element.strip())
                    else:
                        element_set.add(element.strip())


    for data in ['NYT10', 'tacred', 'crossRE', 'FewRel']:
        print(data)
        with open(f'/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE/{data}/test.jsonl', "r") as f:
            for line in f.read().splitlines():
                sample = json.loads(line)
                triple_list = []
                for triple in sample['relationMentions']:
                    relation = triple['label'].replace('_', ' ').replace('/', ' ').replace("-", " ").strip()
                    triple_list.append([triple['em1Text'].lower(), relation.lower(), triple['em2Text'].lower()])
                get_elements(triple_list, embd_list)

        for model in ["OpenAI/gpt-4o-mini", "openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407",
                      "google/gemma-2-9b-it"]:
            print(model)
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
                            if sample['pred_label']:
                                get_elements(sample['pred_label'], embd_list)

    batches = []
    for element in element_set:
        batches.append({"custom_id": element, "method": "POST", "url": "/v1/embeddings",
     "body": {"model": "text-embedding-3-large", "input": element}})

    batch_size = 40000
    outpath = f'{args.base_path}/embd_batches-v1'
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