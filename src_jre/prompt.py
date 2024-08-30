import random


def get_demo(prompt, demo_list, entity_info):
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
                triple_str += f'({subj}:{subj_type},{relation},{obj}:{obj_type})'
            else:
                triple_str += f'({subj}:{subj_type},{relation},{obj}:{obj_type})'
        triple_str += "]"
        prompt += f"Context: {sentence}\nOutput: {triple_str}"
    return prompt


def create_prompt(args, input, demo_list, data_processor):
    messages = []
    if args.entity_info:
        triple_format = '(subject:subject_type, relation, object:object_type)'
    else:
        triple_format = '(subject, relation, object)'

    prompt = ("You are an expert of entity and relation extraction.\n## Goals: Extract entity relation "
              "triples based on the given context. The entity relation schema should be extracted in the form of"
              f" {triple_format}. If more than one triple exists you can output a list of triples such as [{triple_format}, {triple_format}]\n")

    # if args.entity_info:
    #     prompt+=f"Possible entity types:{}\n"
    if args.rel_info:
        prompt += f"Possible relation types:{list(data_processor.rel2prompt.values())}\n"

    prompt += "Please refer to the examples I provided for details.\n## Examples:\n"

    # prompt = ("Suppose you are an entity-relationship triple extraction model. "
    #           "I’ll give you list of head entity types: subject_types, "
    #           "list of tail entity types: object_types, list of relations: relations. "
    #           "Given a sentence, please extract the subject and object in the sentence based on these three lists, "
    #           f"and form a triplet in the form of {triple_format}.")

    prompt = get_demo(prompt, demo_list, args.entity_info)


    testsen = input.sentence
    # prompt += f"## Input:\nContext: {testsen}\nOutput:"

    prompt += f"\nContext: {testsen}\nIn the sentence, what triples might be contained? Please answer in the form {triple_format}:"
    messages.append({"role": "user", "content": prompt})
    return messages
