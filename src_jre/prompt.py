import random

def random_demo_ret(args, prompt, demo_dict, rel2prompt):
    pass
    # for rel in demo_dict.keys():
    #     random.shuffle(demo_dict[rel])
    #     kshot = demo_dict[rel]
    #     for data in kshot:
    #
    #         if "schema" in args.prompt:
    #             prompt += "Context: " + sentence + " The relation between " + headtype + " '" + head + "' and " + tailtype + " '" + tail + "' in the context is " + relation + ".\n"
    #         else:
    #             prompt += "Context: " + sentence + " The relation between '" + head + "' and '" + tail + "' in the context is " + relation + ".\n"
    # return prompt

def knn_demo_ret(prompt, demo_list, entity_info):
    for demo in demo_list:
        sentence = demo.sentence

        triple_str = "["
        for triple in demo.triples:
            subj = triple['head']
            obj = triple['tail']
            subj_type = triple['head_type']
            obj_type = triple['tail_type']
            relation = triple['prompt_relation']
            if entity_info:
                triple_str+=f'({subj}:{subj_type},{relation},{obj}:{obj_type})'
            else:
                triple_str+=f'({subj}:{subj_type},{relation},{obj}:{obj_type})'
        triple_str+="]"
        prompt+= f"Context: {sentence}\nOutput: {triple_str}"
    return prompt

def create_prompt(args, input, demo_list, data_processor):
    if args.entity_info:
        triple_format = '(subject:subject_type, relation, object:object_type)'
    else:
        triple_format = '(subject, relation, object)'

    prompt = ("## Role:\nYou are an expert of entity and relation extraction.\n## Goals: Extract entity relation "
              "triples based on the given context. The entity relation schema should be extracted in the form of"
              f" {triple_format}.\n")

    # if args.entity_info:
    #     prompt+=f"Possible entity types:{}\n"
    if args.rel_info:
        prompt+=f"Possible relation types:{list(data_processor.rel2prompt.values())}\n"

    prompt+= "Please refer to the examples I provided for details.\n## Examples:\n"

    # prompt = ("Suppose you are an entity-relationship triple extraction model. "
    #           "I’ll give you list of head entity types: subject_types, "
    #           "list of tail entity types: object_types, list of relations: relations. "
    #           "Given a sentence, please extract the subject and object in the sentence based on these three lists, "
    #           f"and form a triplet in the form of {triple_format}.")

    if args.demo == 'knn':
        prompt = knn_demo_ret(prompt, demo_list, args.entity_info)
    elif args.demo_ret == 'random':
        prompt = random_demo_ret(prompt, prompt, demo_list, data_processor.rel2prompt)

    testsen = input.sentence
    # prompt += f"## Input:\nContext: {testsen}\nOutput:"

    prompt += f"\n## Input:\nContext: {testsen}\nIn the sentence, what triples might be contained? Please answer in the form {triple_format}:"
    return prompt