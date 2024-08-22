import json
from simcse import SimCSE
import faiss

def generate_ft_dict(args):
    ft_dict = {}
    knn_dict = {}
    train_dict = {}
    if args.use_dev and args.store_error_reason:
        knn_path = "./knn_ids/knn_ids_{}_train_dev.txt".format(args.task)
    elif args.use_dev:
        knn_path = "./knn_ids/knn_ids_{}_dev.txt".format(args.task)
    else:
        knn_path = "./knn_ids/knn_ids_{}.txt".format(args.task)
    with open(knn_path, "r") as f:
        num_id = 0
        for line in f.read().splitlines():
            knn_num = line.split(" ")
            ft_dict[num_id] = knn_num[:args.k]
            num_id += 1

    with open(args.test_dataset, "r") as f:
        num_id = 0
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            knn_dict[tmp_dict["doc_key"]] = ft_dict[num_id]
            num_id += 1
    with open(args.example_dataset, "r") as f:
        num_id = 0
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            train_dict[num_id] = tmp_dict
            num_id += 1
    knn_ft_dict = {}
    for key in knn_dict.keys():
        #print(knn_dict[key])
        #print(train_dict)
        knn_ft_dict[key] = [train_dict[int(x)] for x in knn_dict[key]]
    return knn_ft_dict

def get_demonstrations(args, example_dict, reltoid):
    '''
    Trains the demonstration models
    :param args:
    :param example_dict:
    :param reltoid:
    :return:
    '''
    ft_dict, gpu_index_flat, train_dict, train_sentences, knn_model = None, None, None, None, None
    train_list = [x for y in example_dict.values() for x in y]
    if args.no_na:
        if args.task == "semeval":
            train_list = [x for x in train_list if reltoid[x["relations"][0][0][4]] != 0]
        else:
            train_list = [x for x in train_list if x["relations"] != [[]]]
    if args.entity_info:
        train_dict = {instance(x).reference:x for x in train_list}
        train_sentences = [instance(x).reference for x in train_list]
    else:
        train_dict = {instance(x).sentence:x for x in train_list}
        train_sentences = [instance(x).sentence for x in train_list]

    knn_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    #knn_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    knn_model.build_index(train_sentences, device="cuda")
    return ft_dict, gpu_index_flat, train_dict, train_sentences, knn_model

def generate_ft_example(tmp_dict, ft_dict, reltoid, idtoprompt, demo, args):
    tmp_example = instance(tmp_dict)

    example_list = ft_dict[tmp_example.id]

    return example_list