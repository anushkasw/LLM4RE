import genres
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def ele_proxy(element):
    return np.zeros(3072).tolist()

def get_triple_embedding(triple):
    entity_emb = np.add(genres.ele_emb_dict[triple[0]], genres.ele_emb_dict[triple[2]])
    triple_emb = np.add(entity_emb, genres.ele_emb_dict[triple[1]])
    # emb_ = np.concatenate([ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[2]]])
    # triple_emb = np.concatenate([emb_, ELE_EMB_DICT[triple[1]]])
    return triple_emb.tolist()


def calculate_uniqueness(vectors, phi=0.95):
    """Calculate the Uniqueness Score using cosine similarity and a threshold."""
    similarity_matrix = cosine_similarity(vectors)
    np.fill_diagonal(similarity_matrix, 1)  # Ignore self-similarity

    # Count pairs with cosine similarity smaller than the threshold
    count_smaller_than_phi = np.sum(similarity_matrix < phi)

    total_pairs = len(vectors) * (len(vectors) - 1)
    return count_smaller_than_phi / total_pairs if total_pairs > 0 else 1


def calculate_uniqueness_for_text(triples):
    """Calculate the uniqueness score for a batch of texts."""
    vectors = []
    for triple in triples:
        try:
            vectors.append(get_triple_embedding(triple))
        except:
            continue

    try:
        return calculate_uniqueness(np.array(vectors))
    except:
        return 1


def calculate_uniqueness_score(tmp_dict, output_all_scores=False):
    """Calculate the Uniqueness Score for a dataset using multi-threading."""
    scores = []
    def process_triples(triples):
        # if no triples, return 0
        if not triples or len(triples) == 0:
            return 1
        return calculate_uniqueness_for_text(triples)

    for idx, dict_ in tmp_dict.items():
        triples = dict_['pred_label']
        us = process_triples(triples)
        scores.append(us)
        tmp_dict[idx]['us'] = us

    avg_score = np.mean(scores)
    if output_all_scores:
        return avg_score, scores
    return avg_score, tmp_dict
