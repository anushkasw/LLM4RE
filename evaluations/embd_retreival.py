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
from data_loader import get_RC_data, get_JRE_data

# if not os.path.exists('/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict.pkl'):
#     embd_dict = None
# else:
#     with open('/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict.pkl', 'rb') as handle:
#         embd_dict = pickle.load(handle)

def get_embd_list():
    embd_dict = {}
    files = list(
        Path(f'/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/embd_batches/input'
             ).rglob('*.jsonl'))

    for file in files:
        with open(file) as f:
            res_dict = f.read().splitlines()
        res_dict = [json.loads(line) for line in res_dict if line != '']

        for res in res_dict:
            embd_dict[res['custom_id']] = res
    return embd_dict


def main(args):
    # embd_list = get_embd_list()
    embd_dict = {}
    element_set = set()

    def get_elements(triple_list, embd_dict):
        # if len(triple_list) == 1: #for RC
        #     element = triple_list[0].strip()
        #     if embd_dict:
        #         if element not in embd_dict:
        #             element_set.add(element)
        #     else:
        #         element_set.add(element)
        # else:
        for triple in triple_list:
            if type(triple[0]) == list:
                for triple_ in triple:
                    for element in triple_:
                        if embd_dict:
                            if element.strip() not in embd_dict:
                                element_set.add(element.strip())
                        else:
                            element_set.add(element.strip())
            else:
                for element in triple:
                    if embd_dict:
                        if element.strip() not in embd_dict:
                            element_set.add(element.strip())
                    else:
                        element_set.add(element.strip())


    for data in ['NYT10', 'tacred', 'crossRE', 'FewRel']:
        print(data)
        if args.exp == 'JRE':
            data_dict, _ = get_JRE_data(data, '/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE')
        else:
            data_dict, _ = get_RC_data(data, '/home/UFAD/aswarup/research/Relation-Extraction/Data')

        for dict_ in data_dict.values():
            if args.exp == 'JRE':
                triple_list = []
                pred_list = dict_['triples']
                for triple in pred_list:
                    if len(triple[0]) > 0:
                        lst = list(triple)
                        lst[1] = lst[1].replace('_', ' ').replace('/', ' ').replace("-", " ").strip()
                        triple = tuple(lst)
                        triple_list.append(triple)
            else:
                triple_list = [dict_['relation'].replace('_', ' ').replace('/', ' ').replace("-", " ").strip()]
            get_elements(triple_list, embd_dict)

        for model in ["OpenAI/gpt-4o-mini", "openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407",
                      "google/gemma-2-9b-it"]:
            print(model)
            files = list(
                Path(f'{args.base_path}/processed_results/{args.exp}/{data}/{model}'
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
                                get_elements(sample['pred_label'], embd_dict)

    if len(element_set)>0:
        batches = []
        for element in element_set:
            batches.append({"custom_id": element, "method": "POST", "url": "/v1/embeddings",
         "body": {"model": "text-embedding-ada-002", "input": element}})

        batch_size = 40000
        outpath = f'{args.base_path}/embd_batches-text-ada'
        os.makedirs(outpath, exist_ok=True)
        for i in range(0, len(batches), batch_size):
            batch = batches[i:i + batch_size]
            with open(f'{outpath}/input_{i+1}-{i + batch_size}.jsonl', 'w') as file:
                for item in batch:
                    json.dump(item, file)
                    file.write('\n')
    else:
        print('No new elements found!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="JRE")
    parser.add_argument('--ts', type=bool, default=False)
    # parser.add_argument('--model_name', '-m', type=str, required=False, help="Model Name.", default="mistral")
    #
    parser.add_argument('--base_path', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25")
    # parser.add_argument("--output_dir", default='./output', type=str, required=False,
    #                     help="The output directory where the lda model")

    args = parser.parse_args()
    main(args)