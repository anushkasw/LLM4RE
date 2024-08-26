import json
import re

def flatten_list(labels):
    flattened = []
    for item in labels:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

class DataProcessor:
    def __init__(self, cfg):
        with open(f'{cfg.data_dir}/{cfg.task}/rel2id.json', "r") as f:
            self.rel2id = json.loads(f.read())

        # Mapping 'no_relation' and 'Other' labels to 'NONE'
        if cfg.task in ["semeval_nodir", "GIDS"]:
            self.rel2id['NONE'] = self.rel2id.pop('Other')
            cfg.na_idx = self.rel2id['NONE']
        elif cfg.task in ["tacred", "tacrev", "retacred", "dummy_tacred", "kbp37_nodir"]:
            self.rel2id['NONE'] = self.rel2id.pop('no_relation')
            cfg.na_idx = self.rel2id['NONE']

        self.rel2prompt = self.get_rel2prompt(cfg)

        # Load paths for train and test datasets
        if cfg.demo == 'sel_adapt':
            self.train_path = f'{cfg.data_dir}/{cfg.task}/train.json'
        self.test_path = f'{cfg.data_dir}/{cfg.task}/test.json'

    def get_train_examples(self):
        return self.flatten_examples(self.get_examples(self.train_path))

    def get_test_examples(self):
        return self.flatten_examples(self.get_examples(self.test_path))

    def get_examples(self, example_path):
        example_dict = {k:list() for k in self.rel2id.values()}
        with open(example_path, "r") as f:
            for line in f.read().splitlines():
                tmp_dict = json.loads(line)
                for dict_ in tmp_dict:
                    if dict_["relation"] in ['no_relation', 'Other']:
                        rel = "NONE"
                        dict_["relation"] = "NONE"
                    else:
                        rel = dict_["relation"]
                    example_dict[self.rel2id[rel]].append(dict_)
        return example_dict

    def flatten_examples(self, example_dict):
        # Flatten the examples to be a single list of dictionaries
        flattened_examples = []
        for examples in example_dict.values():
            flattened_examples.extend(examples)
        return flattened_examples

    def get_rel2prompt(self, cfg):
        rel2prompt = {}
        for name, id in self.rel2id.items():
            if cfg.task == 'wiki80':
                labels = name.split(' ')

            elif cfg.task == 'semeval_nodir':
                labels = name.split('-')

            elif cfg.task == 'FewRel':
                labels = name.split('_')

            elif cfg.task in ['NYT10', 'GIDS']:
                if name == 'Other':
                    labels = ['None']
                elif name == '/people/person/education./education/education/institution':
                    labels = ['person', 'and', 'education', 'institution']
                elif name == '/people/person/education./education/education/degree':
                    labels = ['person', 'and', 'education', 'degree']
                else:
                    labels = name.split('/')
                    labels[-1] = "and_"+labels[-1]
                    labels = labels[2:]
                    for idx, lab in enumerate(labels):
                        if "_" in lab:
                            labels[idx] = lab.split("_")
                    labels = flatten_list(labels)

            elif cfg.task == 'WebNLG':
                name_mod = re.sub(r"['()]", '', name)
                labels = name_mod.split(' ')

                if len(labels) == 1:
                    label0 = labels[0]
                    if "_" in label0:
                        labels = label0.split("_")

                        for idx, lab in enumerate(labels):
                            if any(char.isupper() for char in lab) and not lab.isupper():
                                l = re.split(r'(?=[A-Z])', lab)
                                if l[0] == "":
                                    l = l[1:]
                                labels[idx] = l

                        labels = flatten_list(labels)

                    elif any(char.isupper() for char in label0):
                        labels = re.split(r'(?=[A-Z])', label0)

            elif cfg.task == 'crossRE':
                if name == "win-defeat":
                    labels = ['win', 'or', 'defeat']
                else:
                    labels = name.split('-')

            elif cfg.task in ['tacred', 'tacrev', 'retacred', 'dummy_tacred', 'kbp37']:
                labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                            "organization").replace(
                    "stateor", "state or ")]

            labels = [item.lower() for item in labels]

            if cfg.task == 'semeval_nodir':
                rel2prompt[name] = ' and '.join(labels).upper()
            else:
                rel2prompt[name] = ' '.join(labels).upper()
        return rel2prompt
