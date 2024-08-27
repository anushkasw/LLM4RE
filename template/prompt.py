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

def knn_demo_ret(prompt_type, prompt, demo_list, exp):
    for demo in demo_list:
        sentence = demo['sentence']
        subj = demo['head']
        obj = demo['tail']
        subj_type = demo['head_type']
        obj_type = demo['tail_type']
        relation = demo['prompt_label']
        if exp == 'rc':
            if "schema" in prompt_type:
                prompt += "Context: " + sentence + " The relation between " + subj_type + " '" + subj + "' and " + obj_type + " '" + obj + "' in the context is " + relation + ".\n"
            else:
                prompt += "Context: " + sentence + " The relation between '" + subj + "' and '" + obj + "' in the context is " + relation + ".\n"
        elif exp == 'jre':
            prompt+= f"# content:\n{sentence}\n# output:\n[]"

    return prompt

def create_prompt(args, input, demo_list, data_processor):
    if args.exp=='rc':
        if "text" in args.prompt:
            prompt = "There are candidate relations: " + ', '.join(data_processor.rel2prompt.values()) + ".\n"
        else:
            prompt = "Given a context, a pair of head and tail entities in the context, decide the relationship between the head and tail entities from candidate relations: " + \
                        ', '.join(data_processor.rel2prompt.values()) + ".\n"
    elif args.exp=='jre':
        prompt = "## Role:\nYou are an expert of entity and relation extraction.\n## Goals: Extract entity relation triples based on the given content. The entity relation schema should be extracted in the form of [(entity1, relation, entity2)].\nPlease refer to the examples I provided for details.\n## Examples:\n"

    if args.demo == 'knn':
        prompt = knn_demo_ret(args.prompt, prompt, demo_list, args.exp)
    elif args.demo_ret == 'random':
        prompt = random_demo_ret(args.prompt, prompt, demo_list, data_processor.rel2prompt)

    testsen = input['sentence']
    testhead = input['head']
    testtail = input['tail']
    testheadtype = input['head_type']
    testtailtype = input['tail_type']

    if "schema" in args.prompt:
        prompt += "Context: " + testsen + " The relation between " + testheadtype + " '" + testhead + "' and " + testtailtype + " '" + testtail + "' in the context is "
    else:
        prompt += "Context: " + testsen + " The relation between '" + testhead + "' and '" + testtail + "' in the context is "
    return prompt