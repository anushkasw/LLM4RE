import random


def clean(text):
    text = text.lower()
    text = text.replace('  ', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', '')
    return text

def get_demo(demo_list, reason=None):
    demo_prompt = ''
    for demo in demo_list:
        sentence = demo.sentence

        triple_str = "["
        for triple in demo.triples:
            subj = triple['head'].upper()
            obj = triple['tail'].upper()
            relation = triple['prompt_relation'].upper()
            triple_str += f'[{subj}, {relation}, {obj}]'
        triple_str += "]"
        demo_prompt += f"Context: {sentence}\nGiven the context, the entity and relation triplets are: {triple_str}\n"

        if reason:
            demo_prompt += f"Reason:\n"
            for triple in demo.triples:
                demo_prompt += f"{clean(reason[triple['id']])}\n"
    return demo_prompt

def create_prompt(args, input, demo_list, data_processor):
    if not args.demo == 'zero':
        with open(f'{args.prompt_dir}/JRE_{args.prompt}.txt', 'r') as f:
            prompt = f.read()
    else:
        with open(f'{args.prompt_dir}/JRE_{args.prompt}_0.txt', 'r') as f:
            prompt = f.read()

    if args.prompt == 'entrel':
        relations = list(data_processor.rel2prompt.values())
        entities = list(data_processor.ner2id.keys())
        prompt = prompt.replace("$RELATION_SET$", '[' + ', '.join(str(x) for x in relations) + ']')
        prompt = prompt.replace("$ENTITY_SET$", '[' + ', '.join(str(x) for x in entities) + ']')

    if not args.demo == 'zero':
        examples = get_demo(demo_list, data_processor.reasons)
        prompt = prompt.replace("$EXAMPLES$", examples)

    testsen = input.sentence
    prompt = prompt.replace("$TEXT$", testsen)
    messages = [{"role": "user", "content": prompt}]
    return messages