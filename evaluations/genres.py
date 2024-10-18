import argparse
import os
import json
import pickle
import joblib
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


def main(args):
    global ele_emb_dict

    with open(args.embd_path,
              'rb') as handle:
        ele_emb_dict = pickle.load(handle)

    if args.exp == 'JRE':
        data_dict, rel2id = get_JRE_data(args.data, args.data_dir)
    else:
        data_dict, rel2id = get_RC_data(args.data, args.data_dir)

    rel2prompt = get_rel2prompt(args.data, rel2id)

    prompt2rel = {val: key for key, val in rel2prompt.items()}

    for idx, sample in data_dict.items():
        triples = sample['triples']
        for i, triple in enumerate(triples):
            if len(triple) > 1:
                triple = list(triple)
                triple[1] = rel2prompt[triple[1]]
            triples[i] = tuple(triple)

    dictionary = joblib.load(
        open(f'{args.base_path}/topical_models/{args.data}/dictionary.joblib', 'rb'))
    lda_model = joblib.load(
        open(f'{args.base_path}/topical_models/{args.data}/lda.joblib', 'rb'))

    files = list(
        Path(f'{args.base_path}/processed_results/{args.exp}/{args.data}/{args.model}'
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

        output_file = f'{args.base_path}/{args.out_dir}/JRE/{args.data}/{args.model}/{demo}/seed-{seed}'
        os.makedirs(output_file, exist_ok=True)

        check = sanity_check(args.exp, dataset, prompt)
        if not os.path.exists(f'{output_file}/{file.name}'):
            if check:
                tmp_dict = {}
                with open(file, "r") as f:
                    for line in f.read().splitlines():
                        sample = json.loads(line)
                        tmp_dict[sample['id']] = sample

                # ts, tmp_dict = get_ts_scores(args.exp, data_dict, tmp_dict, dictionary, lda_model)
                us, tmp_dict = calculate_uniqueness_score(tmp_dict)
                cs, tmp_dict = calculate_completeness_score(tmp_dict, data_dict, rel2prompt)
                with open(f'{output_file}/{file.name}', 'w') as f:
                    for tmp in tmp_dict.items():
                        if f.tell() > 0:  # Check if file is not empty
                            f.write('\n')
                        json.dump(tmp, f)
                print(f'File saved in: {output_file}/{file.name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="JRE")
    parser.add_argument('--data', '-d', type=str, required=False, help="Dataset Type", default="FewRel")
    parser.add_argument('--model', '-m', type=str, required=False, help="Model Type", default="google/gemma-2-9b-it")

    parser.add_argument('--base_path', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25")
    parser.add_argument('--embd_path', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/gre_element_embedding_dict_small.pkl")
    parser.add_argument('--out_dir', '-dir', type=str, required=False,
                        default="genres_metrics")
    parser.add_argument("--data_dir", default='/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE', type=str,
                        required=False,
                        help="raw data dir")

    args = parser.parse_args()

    main(args)
