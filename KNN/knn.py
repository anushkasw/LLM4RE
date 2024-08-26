import json
from simcse import SimCSE

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

    knn_result = model.search(test_sentences, device="cuda", threshold=0.0, top_k=k)
    knn_list = [train_dict[x[0]] for x in knn_result]
    return knn_list