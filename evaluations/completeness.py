from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from time import sleep

import genres
from openai import OpenAI
client = OpenAI(api_key="sk-proj-5ysoQyEaYVkjhNgpPU62T3BlbkFJD3nz6JagSAF8sNwUwN8f")


def embedding_retriever(term):
    print('get embds')
    i = 0
    while (i < 5):
        try:
            # Send the request and retrieve the response
            response = client.embeddings.create(
                input=str(term),
                model="text-embedding-3-large"
            )

            # Extract the text embeddings from the response JSON
            embedding = response.data[0].embedding

            return embedding

        except Exception as e:
            i += 1
            print(f"Error in gpt_instruct: \"{term}\". Retrying...")
            sleep(5)

    return np.zeros(3072).tolist()


def ele_proxy(element):
    return np.zeros(3072).tolist()


def check_triple(ELE_EMB_DICT, element):
    if element not in ELE_EMB_DICT:
        ELE_EMB_DICT[element] = embedding_retriever(element)
    return ELE_EMB_DICT

def get_gt_embds(gt_text_triples):
    gt_triple_emb_store = {}
    gt_relation_emb_store = {}
    for idx in gt_text_triples.keys():
        gt_triple_list = gt_text_triples[idx]['triples']
        for triple in gt_triple_list:
            triple_str = str(triple)
            check_triple(genres.ele_emb_dict, triple[0])
            check_triple(genres.ele_emb_dict, triple[1])
            check_triple(genres.ele_emb_dict, triple[2])

            entity_emb = np.add(genres.ele_emb_dict[triple[0]], genres.ele_emb_dict[triple[2]])
            triple_emb = np.add(np.array(entity_emb), np.array(genres.ele_emb_dict[triple[1]]))
            # emb_ = np.concatenate([genres.ele_emb_dict[triple[0]], genres.ele_emb_dict[triple[1]]])
            # triple_emb = np.concatenate([emb_, genres.ele_emb_dict[triple[2]]])
            gt_triple_emb_store[triple_str] = triple_emb.tolist()
            gt_relation_emb_store[triple_str] = genres.ele_emb_dict[triple[1]]
    return gt_triple_emb_store, gt_relation_emb_store


def calculate_completeness_score(tmp_dict, data_dict, model_name=None, threshold=0.95,
                                 output_all_scores=False):
    gt_triple_emb_store, gt_relation_emb_store = get_gt_embds(data_dict)
    completeness_scores = []
    scores_details = defaultdict(dict)
    text2cs = {}

    for idx, dict_ in tmp_dict.items():
        gt_triples = data_dict[idx]['triples']
        triples = dict_['pred_label']

        if not triples or len(triples) == 0:
            tmp_dict[idx]['cs'] = 0
            completeness_scores.append(0)
            continue

        if len(gt_triples) == 0:
            tmp_dict[idx]['cs'] = 1
            completeness_scores.append(1)
            continue

        # if type(triples[0][0]) == list:
        #     triples = triples[0]
        # else:
        #     triples = triple

        gt_embeddings = {str(triple): gt_triple_emb_store[str(triple)] for triple in gt_triples}
        # Recall calculation
        gt_recalls = {gt_triple: 0 for gt_triple in gt_embeddings.keys()}

        extracted_triple_embeddings = []
        extracted_relation_embeddings = []
        for triple in triples:
            try:
                entity_emb = np.add(genres.ele_emb_dict[triple[0]], genres.ele_emb_dict[triple[2]])
                triple_emb = np.add(entity_emb, genres.ele_emb_dict[triple[1]])
                # emb_ = np.concatenate([ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[1]]])
                # triple_emb = np.concatenate([emb_, ELE_EMB_DICT[triple[2]]])
                extracted_triple_embeddings.append(triple_emb.tolist())
                extracted_relation_embeddings.append(genres.ele_emb_dict[triple[1]])
            except:
                continue
        if len(extracted_triple_embeddings) == 0:
            continue
        for gt_triple, gt_embedding in gt_embeddings.items():
            similarity_scores = cosine_similarity([gt_embedding], extracted_triple_embeddings)
            best_match_score = np.max(similarity_scores)
            best_match_index = np.argmax(similarity_scores)
            if best_match_score >= threshold:
                if model_name == 'gpt-3.5_closed':
                    extracted_relation_emb = extracted_relation_embeddings[best_match_index]
                    relation_similarity_score = cosine_similarity([gt_relation_emb_store[gt_triple]],
                                                                  [extracted_relation_emb])
                    if relation_similarity_score >= threshold:
                        gt_recalls[gt_triple] = 1
                else:
                    gt_recalls[gt_triple] = 1

            # Store details
            # scores_details[text][gt_triple] = similarity_scores.tolist()[0]

        # Compute completeness score for this text
        score = sum(gt_recalls.values()) / len(gt_recalls) if len(gt_recalls) > 0 else 0
        completeness_scores.append(score)
        tmp_dict[idx]['cs'] = score
        # text2cs[text] = score

    avg_completeness_score = np.mean(completeness_scores) if completeness_scores else 0

    return avg_completeness_score, tmp_dict
