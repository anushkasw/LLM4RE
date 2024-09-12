
def calculate_metrics(predicted_relations, ground_truth_relations):
        prec, rec = 0, 0

        # Count correct predictions
        for pred in predicted_relations:
            if pred in ground_truth_relations:
                prec += 1

        for gt in ground_truth_relations:
            if gt in predicted_relations:
                rec += 1

        precision = prec / len(predicted_relations)
        recall = rec / len(ground_truth_relations)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1

def get_traditional_scores(tmp_dict, prompt2rel):
    precision = []
    recall = []
    f1 = []

    for idx, dict_ in tmp_dict.items():
        triples = dict_['pred_label']
        for trip in triples:
            if len(trip) > 1:
                if trip[1] in prompt2rel:
                    trip[1] = prompt2rel[trip[1]]

    for idx, dict_ in tmp_dict.items():
        gt_triples = dict_['true_label']
        try:
            pred_triple_str = [" ".join(triple).lower() for triple in dict_['pred_label']]
            gt_triple_str = [" ".join(triple).lower() for triple in gt_triples]
            p, r, f = calculate_metrics(pred_triple_str, gt_triple_str)
            precision.append(p)
            recall.append(r)
            f1.append(f)
        except:
            continue

    return (sum(precision) / len(precision)), (sum(recall) / len(recall)), (sum(f1) / len(f1))