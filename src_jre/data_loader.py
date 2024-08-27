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

class instance:
    def __init__(self, tmp_dict, rel2prompt):
        self.id = tmp_dict["sample_id"]
        self.sentence = tmp_dict['sentText']

        self.triples = []
        for mentions in tmp_dict['relationMentions']:
            if mentions["label"] in ['no_relation', 'Other']:
                relation = "NONE"
            else:
                relation = mentions["label"]
            prompt_label = rel2prompt[relation]

            if mentions['em1Type'] == "misc":
                headtype = "miscellaneous"
            elif mentions['em1Type'] == 'O':
                headtype = "unkown"
            else: headtype = mentions['em1Type']

            if mentions['em2Type'] == "misc":
                tailtype = "miscellaneous"
            elif mentions['em2Type'] == 'O':
                tailtype = "unkown"
            else: tailtype = mentions['em2Type']

            self.triples.append({
                'head': mentions['em1Text'],
                'tail': mentions['em2Text'],
                'head_type': headtype,
                'tail_type': tailtype,
                'relation': relation,
                'prompt_relation': prompt_label
            }
            )

        # self.reference = ("The relation between \"" + self.head + "\" and \""
        #                   + self.tail + "\" in the sentence \"" + self.sentence + "\"")
        # self.context = "\nContext: " + self.sentence
        # self.query = ("\nQuestion: What is the relation between " + self.head
        #               + " and " + self.tail + "?")
        # self.clue = "\nClues: "
        # self.pred = "\nAnswer: "
        # self.prompt = self.context + self.query

    # def get_relation(self, tmp_dict):
    #     if tmp_dict["relations"] == [[]]:
    #         return "NONE"
    #     else:
    #         return tmp_dict["relations"][0][0][4]
    #
    # def get_reason(self, idtoprompt, reltoid):
    #     reason = ("What are the clues that lead the relation between \""
    #               + self.head + "\" and \"" + self.tail + "\" to be "
    #               + idtoprompt[reltoid[self.rel]] + " in the sentence \"" + string + "\"?")
    #     return reason

class DataProcessor:
    def __init__(self, args):
        with open(f'{args.data_dir}/{args.task}/rel2id.json', "r") as f:
            self.rel2id = json.loads(f.read())

        # Mapping 'no_relation' and 'Other' labels to 'NONE'
        if args.task in ["semeval_nodir", "GIDS"]:
            self.rel2id['NONE'] = self.rel2id.pop('Other')
            args.na_idx = self.rel2id['NONE']
        elif args.task in ["tacred", "tacrev", "retacred", "dummy_tacred", "kbp37_nodir"]:
            self.rel2id['NONE'] = self.rel2id.pop('no_relation')
            args.na_idx = self.rel2id['NONE']

        self.rel2prompt = self.get_rel2prompt(args)

        # Demonstration Retrieval
        self.train_path = f'{args.data_dir}/{args.task}/train.jsonl'
        self.test_path = f'{args.data_dir}/{args.task}/test.jsonl'
        # self.test_path = f'{args.data_dir}/{args.task}/k-shot/seed-{args.data_seed}/test.jsonl' # TODO: reorg test data

    def get_train_examples(self):
        return self.get_examples(self.train_path)

    def get_test_examples(self):
        return self.get_examples(self.test_path)

    def get_examples(self, example_path):
        example_dict = {}
        with open(example_path, "r") as f:
            for line in f.read().splitlines():
                tmp_dict = json.loads(line)
                example_dict[tmp_dict['sample_id']] = instance(tmp_dict, self.rel2prompt)
        return example_dict

    def get_rel2prompt(self, args):
        rel2prompt = {}
        for name, id in self.rel2id.items():
            if args.task == 'wiki80':
                labels = name.split(' ')

            elif args.task == 'semeval_nodir':
                labels = name.split('-')

            elif args.task == 'FewRel':
                labels = name.split('_')

            elif args.task in ['NYT10', 'GIDS']:
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

            elif args.task == 'WebNLG':
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

            elif args.task == 'crossRE':
                if name == "win-defeat":
                    labels = ['win', 'or', 'defeat']
                else:
                    labels = name.split('-')

            elif args.task in ['tacred', 'tacrev', 'retacred', 'dummy_tacred', 'kbp37']:
                labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                            "organization").replace(
                    "stateor", "state or ")]

            labels = [item.lower() for item in labels]

            if args.task == 'semeval_nodir':
                rel2prompt[name] = ' and '.join(labels).upper()
            else:
                rel2prompt[name] = ' '.join(labels).upper()
        return rel2prompt
