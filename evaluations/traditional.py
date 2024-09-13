
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

def f1_score(true, pred_result):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            if golden not in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable', 'NONE']:
                correct_positive += 1
        if golden not in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable', 'NONE']:
            gold_positive +=1
        if pred_result[i] not in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable', 'NONE']:
            pred_positive += 1
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    return micro_p, micro_r, micro_f1


def f1_score_na(true, pred_result):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            correct_positive += 1
        gold_positive +=1
        pred_positive += 1
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    return micro_p, micro_r, micro_f1


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
        p, r, f = f1_score(pred_label, true_label)
        return p, r, f




