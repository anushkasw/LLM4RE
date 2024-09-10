import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
from collections import defaultdict
import math
import nltk
import json
import pickle
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]


def calculate_ts_score(data, dictionary, lda, output_all_scores=False):
    all_ts_scores = {}
    for source_text in data.keys():
        triples = data[source_text]
        triples_str = ''
        for triple in triples:
            triples_str += f"{triple[0]} {triple[1]} {triple[2]} ."
        processed_source = preprocess(source_text)
        processed_triples = preprocess(triples_str)
        source_corpus = dictionary.doc2bow(processed_source)
        triples_corpus = dictionary.doc2bow(processed_triples)

        source_dist = lda.get_document_topics(source_corpus, minimum_probability=0)
        triples_dist = lda.get_document_topics(triples_corpus, minimum_probability=0)

        ts_score = math.exp(-kullback_leibler(source_dist, triples_dist))
        all_ts_scores[source_text] = ts_score

    average_ts_score = sum(all_ts_scores.values()) / len(all_ts_scores)

    if output_all_scores:
        return average_ts_score, all_ts_scores

    return average_ts_score


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', type=str, default='false')
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--dataset', type=str, default='nyt10m')
    parser.add_argument('--exp_id', type=str, default='1')

    args = parser.parse_args()
    return args


def main(args):
    file_to_evaluate = f'{args.data_dir}/GenRES_results/{args.task}/{args.task}_rand_500_{args.model_name}_{args.exp_id}.json'
    with open(file_to_evaluate, 'r') as f:
        data_to_evaluate = json.load(f)
    dictionary = pickle.load(open(f'{args.data_dir}/GenRES_results/{args.task}/topical_process/{args.task}_dictionary.pkl', 'rb'))
    lda_model = pickle.load(open(f'{args.data_dir}/GenRES_results/{args.task}/topical_process/{args.task}_lda.pkl', 'rb'))

    print(f"Calculating TS score for model {args.model_name} on dataset {args.task}...")
    ts_score = calculate_ts_score(data_to_evaluate, dictionary, lda_model)
    print(f"TS score for model {args.model_name} on dataset {args.task}: {ts_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, required=False, help="Dataset Name.", default="nyt10m")
    parser.add_argument('--model_name', '-m', type=str, required=False, help="Model Name.", default="mistral")

    parser.add_argument('--data_dir', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/ICL")
    parser.add_argument("--output_dir", default='./output', type=str, required=False,
                        help="The output directory where the lda model")

    args = parser.parse_args()
    main(args)



