import json


def get_RC_data(data):
    with open(f'/home/UFAD/aswarup/research/Relation-Extraction/Data/{data}/test.json', "r") as f:
        test_data = json.load(f)

    data_dict = {}
    for sample in test_data:
        row = {
            'text': " ".join(sample['token']).lower(),
            'relation': sample['relation']
        }
        data_dict[sample['id']] = row

    with open(f'/home/UFAD/aswarup/research/Relation-Extraction/Data/{data}/rel2id.json', "r") as f:
        rel2id = json.load(f)

    return data_dict, rel2id

def get_JRE_data(data):
    data_dict = {}
    with open(f'/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE/{data}/test.jsonl', "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            triples = []
            for triple in tmp_dict['relationMentions']:
                triples.append((triple['em1Text'].lower(), triple['label'].lower(), triple['em2Text'].lower()))
            row = {
                'text': tmp_dict['sentText'],
                'triples': triples
            }
            data_dict[tmp_dict['sample_id']] = row

    with open(f'/home/UFAD/aswarup/research/Relation-Extraction/Data_JRE/{data}/rel2id.json', "r") as f:
        rel2id = json.load(f)
    return data_dict, rel2id