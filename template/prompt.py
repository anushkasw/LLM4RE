import random

def random_demo_ret(args, prompt, demo_dict, rel2prompt):
    for rel in demo_dict.keys():
        random.shuffle(demo_dict[rel])
        kshot = demo_dict[rel]
        for data in kshot:
            ss, se = data['subj_start'], data['subj_end']
            head = ' '.join(data['token'][ss:se + 1])
            headtype = data['subj_type'].lower().replace('_', ' ')
            if headtype == "misc":
                headtype = "miscellaneous"
            elif headtype == 'O':
                headtype = "unkown"
            os, oe = data['obj_start'], data['obj_end']
            tail = ' '.join(data['token'][os:oe + 1])
            tailtype = data['obj_type'].lower().replace('_', ' ')
            if tailtype == "misc":
                tailtype = "miscellaneous"
            elif tailtype == 'O':
                tailtype = "unkown"
            sentence = ' '.join(data['token'])
            relation = rel2prompt[data['relation']]
            if "schema" in args.prompt:
                prompt += "Context: " + sentence + " The relation between " + headtype + " '" + head + "' and " + tailtype + " '" + tail + "' in the context is " + relation + ".\n"
            else:
                prompt += "Context: " + sentence + " The relation between '" + head + "' and '" + tail + "' in the context is " + relation + ".\n"
    return prompt

def create_prompt(args, input, demo_dict, data_processor):
    if "text" in args.prompt:
        prompt = "There are candidate relations: " + ', '.join(data_processor.rel2prompt.values()) + ".\n"
    else:
        prompt = "Given a context, a pair of head and tail entities in the context, decide the relationship between the head and tail entities from candidate relations: " + \
                    ', '.join(data_processor.rel2prompt.values()) + ".\n"

    prompt = random_demo_ret(args, prompt, demo_dict, data_processor.rel2prompt)
    tss, tse = input['subj_start'], input['subj_end']
    testhead = ' '.join(input['token'][tss:tse + 1])
    testheadtype = input['subj_type'].lower().replace('_', ' ')
    if testheadtype == "misc":
        testheadtype = "miscellaneous"
    elif testheadtype == 'O':
        testheadtype = "unkown"
    tos, toe = input['obj_start'], input['obj_end']
    testtail = ' '.join(input['token'][tos:toe + 1])
    testtailtype = input['obj_type'].lower().replace('_', ' ')
    if testtailtype == "misc":
        testtailtype = "miscellaneous"
    elif testtailtype == 'O':
        testtailtype = "unkown"
    testsen = ' '.join(input['token'])
    if "schema" in args.prompt:
        prompt += "Context: " + testsen + " The relation between " + testheadtype + " '" + testhead + "' and " + testtailtype + " '" + testtail + "' in the context is "
    else:
        prompt += "Context: " + testsen + " The relation between '" + testhead + "' and '" + testtail + "' in the context is "
    return prompt