import random

def prepare_incontext_sampling(train_samples, 
                                verbalizer,
                                prefix,
                                infix,
                                postfix,
                                ):

    label2token = {v: k for k, v in verbalizer.items()}
    label2samples = {}
    full_samples = []

    for sample in train_samples:
        sentence1 = sample['text']  # Assuming the sentence key is 'text'
        label = sample['relation']  # Assuming relation is the label
        label_token = label2token[label]

        full_sentence = prefix + sentence1 + infix + postfix + label_token
        full_samples.append(full_sentence)

        # empty list if first sample
        label_list = label2samples.get(label, [])
        label_list.append(full_sentence)
        label2samples[label] = label_list

    return label2samples, full_samples

def prepend_incontext_samples(
                                label2samples,
                                full_train_samples,
                                k,
                                balance_sample,
                            ):
    final_sentence = None
    sep = '\n\n\n'

    if k == 0:
        return '', sep

    if balance_sample:
        total_count = 0
        labels = list(label2samples.keys())
        random.shuffle(labels)
        samples_map = {label: [i for i in range(len(label2samples[label]))] for label in labels}
        while True:
            for label in labels:
                samples = label2samples[label]
                total_length = len(samples)
                not_used_indices = [i for i in range(total_length)]
                while True:
                    samples_list = samples_map[label]
                    random_index = random.randint(0, total_length-1)
                    selected_sample = samples[random_index]

                    if final_sentence is None:
                        selected_index = samples_list.index(random_index)
                        samples_list.pop(selected_index)
                        samples_map[label] = samples_list
                        break
                    if random_index in samples_list:
                        selected_index = samples_list.index(random_index)
                        samples_list.pop(selected_index)
                        samples_map[label] = samples_list
                        break

                if final_sentence is None:
                    final_sentence = selected_sample
                else:
                    final_sentence = final_sentence + sep + selected_sample

                total_count += 1
                if total_count == k:
                    return final_sentence, sep
    else:
        full_train_samples_copy = full_train_samples.copy()
        for index in range(k):
            total_length = len(full_train_samples_copy)
            random_index = random.randint(0, total_length-1)
            selected_sample = full_train_samples_copy.pop(random_index)

            if final_sentence is None:
                final_sentence = selected_sample
            else:
                final_sentence = final_sentence + sep + selected_sample

    return final_sentence, sep
