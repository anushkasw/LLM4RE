import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from topical import get_ts_scores
from uniqueness import calculate_uniqueness_score
from traditional import get_traditional_scores
from completeness import calculate_completeness_score

from rel_verbaliser import get_rel2prompt
from utils import sanity_check
from data_loader import get_RC_data, get_JRE_data

# with open('/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict.pkl',
#           'rb') as handle:
#     ELE_EMB_DICT = pickle.load(handle)

with open('/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict.json', 'r') as f:
    ELE_EMB_DICT = json.load(f)

def get_gt_embds(data_dict):
    gt_triple_emb_store = {}
    gt_relation_emb_store = {}
    for key, value in data_dict.items():
        gt_triple_list = value['triples']
        for triple in gt_triple_list:
            triple_str = str(triple)
            entity_emb = np.add(ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[2]])
            triple_emb = np.add(np.array(entity_emb), np.array(ELE_EMB_DICT[triple[1]]))
            # emb_ = np.concatenate([ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[1]]])
            # triple_emb = np.concatenate([emb_, ELE_EMB_DICT[triple[2]]])
            gt_triple_emb_store[triple_str] = triple_emb.tolist()
            gt_relation_emb_store[triple_str] = ELE_EMB_DICT[triple[1]]
    return gt_triple_emb_store, gt_relation_emb_store

def main(args):
    df = pd.DataFrame(columns=['exp', 'dataset', 'model', 'demo', 'seed', 'k', 'prompt', 'f1', 'p', 'r'])

    for data in ['NYT10', 'tacred', 'crossRE', 'FewRel']:
        if args.exp=='jre':
            data_dict, rel2id = get_JRE_data(data)
        else:
            data_dict, rel2id = get_RC_data(data)


        rel2prompt = get_rel2prompt(data, rel2id)
        prompt2rel = {val: key for key, val in rel2prompt.items()}

        dictionary = pickle.load(
            open(f'{args.base_path}/topical_models/{data}/dictionary.pkl', 'rb'))
        lda_model = pickle.load(
            open(f'{args.base_path}/topical_models/{data}/lda.pkl', 'rb'))

        gt_triple_emb_store, gt_relation_emb_store = get_gt_embds(data_dict)

        for model in ["openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407",
                      "google/gemma-2-9b-it", "OpenAI/gpt-4o-mini"]:
            files = list(
                Path(f'{args.base_path}/processed_results/JRE/{data}/{model}'
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

                    # ts = get_ts_scores(data_dict, tmp_dict, dictionary, lda_model) # TODO: fix triples with more than 3 elements
                    uq = calculate_uniqueness_score(tmp_dict, ELE_EMB_DICT)
                    cs = calculate_completeness_score(tmp_dict, gt_triple_emb_store, gt_relation_emb_store, ELE_EMB_DICT)

                    res_dict = tmp_dict.copy()
                    p, r, f1 = get_traditional_scores(res_dict, prompt2rel)

                    row = {'exp': args.exp, 'dataset': dataset, 'model': f'{llm_fam}/{llm}', 'demo': demo, 'seed': seed, 'k': k,
                           'prompt': prompt, 'f1': f1, 'p': p, 'r': r}
                    df.loc[len(df)] = row
    os.makedirs(f'{args.base_path}/eval_csvs', exist_ok=True)
    df.to_csv(f'{args.base_path}/eval_csvs/jre_traditional.csv', index=False)


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
