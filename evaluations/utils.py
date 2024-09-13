
def sanity_check(exp, dataset, prompt):
    check = False
    if exp == 'JRE':
        if dataset in ['crossRE', 'FewRel'] and prompt in ['open']:
            check = True
        elif dataset in ['NYT10', 'tacred'] and prompt in ['open', 'entrel']:
            check = True
    elif exp == 'RC':
        if dataset in ['crossRE', 'FewRel', 'NYT10'] and prompt in ['open', 'rel']:
            check = True
        elif dataset in ['tacred'] and prompt in ['open', 'entrel', 'rel', 'ent']:
            check = True
    return check
