import argparse
import json
import pickle
import pandas as pd
from pathlib import Path
from topical import get_ts_scores
from traditional import get_traditional_scores
from rel_verbaliser import get_rel2prompt
from utils import sanity_check


def main(args):
    for data in ['NYT10']:
        data_dict = {}
        with open(f'{args.base_path}/Data_ICL/data_jre/{data}/test.json', "r") as f:
            for line in f.read().splitlines():
                sample = json.loads(line)
                data_dict[sample['id']] = sample['text']

        with open(f'{args.base_path}/Data_ICL/data_jre/{data}/rel2id.json', "r") as f:
            rel2id = json.load(f)

        rel2prompt = get_rel2prompt(data, rel2id)
        prompt2rel = {val: key for key, val in rel2prompt.items()}

        dictionary = pickle.load(
            open(f'{args.base_path}/topical_models/{data}/dictionary.pkl', 'rb'))
        lda_model = pickle.load(
            open(f'{args.base_path}/topical_models/{data}/lda.pkl', 'rb'))

        for model in ["openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407",
                      "google/gemma-2-9b-it"]:
            files = list(
                Path(f'{args.base_path}/processed_results/JRE/{data}/{model}'
                     ).rglob('*.jsonl'))

            df = pd.DataFrame(columns=['exp', 'dataset', 'model', 'demo', 'seed', 'k', 'prompt', 'f1', 'p', 'r'])
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

                    ts = get_ts_scores(data_dict, tmp_dict, dictionary, lda_model) # TODO: fix triples with more than 3 elements

                    res_dict = tmp_dict.copy()
                    p, r, f1 = get_traditional_scores(res_dict, prompt2rel)

                    row = {'exp': args.exp, 'dataset': dataset, 'model': f'{llm_fam}/{llm}', 'demo': demo, 'seed': seed, 'k': k,
                           'prompt': prompt, 'f1': f1, 'p': p, 'r': r}
                    df.loc[len(df)] = row
    print(df)


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
