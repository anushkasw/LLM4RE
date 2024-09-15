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


def get_ts_scores(exp, data_dict, tmp_dict, dictionary, lda_model):
    all_ts_scores = {}


    for idx, dict_ in tmp_dict.items():
        source_text = data_dict[idx]['text']
        triples = dict_['pred_label']
        triples_str = ''
        if triples and triples != 'Other':
            for triple in triples:
                if len(triple) == 3:
                    triples_str += f"{triple[0]} {triple[1]} {triple[2]} ."
                else:
                    continue
        else:
            continue
        processed_source = preprocess(source_text)
        processed_triples = preprocess(triples_str)
        source_corpus = dictionary.doc2bow(processed_source)
        triples_corpus = dictionary.doc2bow(processed_triples)

        source_dist = lda_model.get_document_topics(source_corpus, minimum_probability=0)
        triples_dist = lda_model.get_document_topics(triples_corpus, minimum_probability=0)

        ts_score = math.exp(-kullback_leibler(source_dist, triples_dist))
        all_ts_scores[source_text] = ts_score
        tmp_dict[idx]['ts'] = ts_score

    average_ts_score = sum(all_ts_scores.values()) / len(all_ts_scores)
    return average_ts_score, tmp_dict
