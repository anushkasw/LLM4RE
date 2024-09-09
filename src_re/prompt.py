import random


def clean(text):
    text = text.lower()
    text = text.replace('  ', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', '')
    return text

def get_demo(demo_list, prompt_type, reason=None):
    demo_prompt = ''
    for demo in demo_list:
        sentence = demo.sentence
        subj = demo.head
        obj = demo.tail
        subj_type = demo.head_type
        obj_type = demo.tail_type
        relation = demo.prompt_label

        if prompt_type == 'ent' or prompt_type == 'entrel':
            demo_prompt += (f"Context: {sentence}\nGiven the context, the relation between {subj} of type {subj_type} "
                            f"and {obj} of type {obj_type} is {relation}\n")
        else:
            demo_prompt += (
                f"Context: {sentence}\nGiven the context, the relation between {subj} and {obj} is {relation}\n")


        if reason:
            demo_prompt += f"Reason:\n"
            demo_prompt += f"{clean(reason[demo['id']])}\n"
    return demo_prompt

def create_prompt(args, input, demo_list, data_processor):
    if not args.demo == 'zero':
        with open(f'{args.prompt_dir}/RC_{args.prompt}.txt', 'r') as f:
            prompt = f.read()
    else:
        with open(f'{args.prompt_dir}/RC_{args.prompt}_0.txt', 'r') as f:
            prompt = f.read()

    relations = list(data_processor.rel2prompt.values())
    if args.prompt == 'rel' or args.prompt == 'entrel':
        prompt = prompt.replace("$RELATION_SET$", '[' + ', '.join(str(x) for x in relations) + ']')

    if not args.demo == 'zero':
        examples = get_demo(demo_list, args.prompt, data_processor.reasons)
        prompt = prompt.replace("$EXAMPLES$", examples)

    testsen = input.sentence
    subj = input.head
    obj = input.tail
    subj_type = input.head_type
    obj_type = input.tail_type

    prompt = prompt.replace("$TEXT$", testsen)
    prompt = prompt.replace("$SUBJECT$", subj)
    prompt = prompt.replace("$OBJECT$", obj)

    if args.prompt == 'ent' or args.prompt == 'entrel':
        prompt = prompt.replace("$SUBJ_TYPE$", subj_type)
        prompt = prompt.replace("$OBJ_TYPE$", obj_type)
    messages = [{"role": "user", "content": prompt}]
    return messages