
def calculate_jre_metrics(predicted_relations, ground_truth_relations):
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

def calculate_rc_metrics(preds, labels, na_idx=False):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        n_pred += 1
        n_gold += 1
        if pred == label:
            n_correct += 1

    if n_correct == 0:
        return 0.0, 0.0, 0.0
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return prec, recall, f1

def get_traditional_scores(exp, tmp_dict, prompt2rel):
    if exp == 'JRE':
        precision = []
        recall = []
        f1 = []

        for idx, dict_ in tmp_dict.items():
            triples = dict_['pred_label']
            if triples:
                for trip in triples:
                    if len(trip) > 1:
                        if trip[1] in prompt2rel:
                            trip[1] = prompt2rel[trip[1]]

        for idx, dict_ in tmp_dict.items():
            gt_triples = dict_['true_label']
            try:
                pred_triple_str = [" ".join(triple).lower() for triple in dict_['pred_label']]
                gt_triple_str = [" ".join(triple).lower() for triple in gt_triples]
                p, r, f = calculate_jre_metrics(pred_triple_str, gt_triple_str)
                precision.append(p)
                recall.append(r)
                f1.append(f)
            except:
                continue
        return (sum(precision) / len(precision)), (sum(recall) / len(recall)), (sum(f1) / len(f1))
    else:
        true_label = []
        pred_label = []
        for idx, dict_ in tmp_dict.items():
            relation = dict_['pred_label']
            if relation in prompt2rel:
                pred_label.append(prompt2rel[relation])
            else:
                pred_label.append(relation)
            true_label.append(dict_['true_label'])
        p, r, f = calculate_rc_metrics(pred_label, true_label, na_idx=True)
        return p, r, f




