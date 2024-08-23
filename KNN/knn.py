import json
from simcse import SimCSE

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

def get_demonstrations(args, example_dict):
    '''
    Trains the demonstration models
    :param args:
    :param example_dict:
    :param reltoid:
    :return:
    '''
    train_dict, knn_model = None, None
    train_list = [x for y in example_dict.values() for x in y]
    if args.no_na:
        train_list = [x for x in train_list if x["relations"] != 'NONE']

    train_dict = {" ".join(x["token"]):x for x in train_list}
    train_sentences = [" ".join(x["token"]) for x in train_list]

    knn_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    #knn_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    knn_model.build_index(train_sentences, device="cuda")
    return train_dict, knn_model


def find_knn_example(model, test_dict, train_dict, k):
    test_sentences = " ".join(test_dict["token"])

    knn_result = model.search(test_sentences, device="cpu", threshold=0.0, top_k=k)
    knn_list = [train_dict[x[0]] for x in knn_result]
    return knn_list