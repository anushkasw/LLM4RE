from gensim.matutils import kullback_leibler
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]


def get_ts_scores(data_dict, tmp_dict, dictionary, lda_model):
    all_ts_scores = {}
    for idx, dict_ in tmp_dict.items():
        source_text = data_dict[idx]['text']
        triples = dict_['pred_label']
        triples_str = ''
        if len(triples) == 3:
            for triple in triples:
                triples_str += f"{triple[0]} {triple[1]} {triple[2]} ."
            processed_source = preprocess(source_text)
            processed_triples = preprocess(triples_str)
            source_corpus = dictionary.doc2bow(processed_source)
            triples_corpus = dictionary.doc2bow(processed_triples)

            source_dist = lda_model.get_document_topics(source_corpus, minimum_probability=0)
            triples_dist = lda_model.get_document_topics(triples_corpus, minimum_probability=0)

            ts_score = math.exp(-kullback_leibler(source_dist, triples_dist))
            all_ts_scores[source_text] = ts_score

    average_ts_score = sum(all_ts_scores.values()) / len(all_ts_scores)
    return average_ts_score


# def main(args):
#     file_to_evaluate = f'{args.data_dir}/GenRES_results/{args.task}/{args.task}_rand_500_{args.model_name}_{args.exp_id}.json'
#     with open(file_to_evaluate, 'r') as f:
#         data_to_evaluate = json.load(f)
#
#
#     print(f"Calculating TS score for model {args.model_name} on dataset {args.task}...")
#     ts_score = calculate_ts_score(data_to_evaluate, dictionary, lda_model)
#     print(f"TS score for model {args.model_name} on dataset {args.task}: {ts_score}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task', '-t', type=str, required=False, help="Dataset Name.", default="NYT10")
#     parser.add_argument('--model_name', '-m', type=str, required=False, help="Model Name.", default="openchat/openchat_3.5")
#
#     parser.add_argument('--result_dir', '-dir', type=str, required=False,
#                         default="/home/UFAD/aswarup/research/Relation-Extraction/LLM4RE/COLING25/processed_results")
#     parser.add_argument("--output_dir", default='./output', type=str, required=False,
#                         help="The output directory where the lda model")
#
#     args = parser.parse_args()
#     main(args)



