import argparse
from datasets import load_dataset
from openicl import DatasetReader
from openicl import TopkRetriever, VotekRetriever, DPPRetriever, MDLRetriever
import json
import os

def main(args):
    print(f'\tDataset={args.task}')
    print(f'\tDemo={args.demo}')
    data_files = {
        'train': f'{args.data_dir}/{args.task}/train.json',
        'test': f'{args.data_dir}/{args.task}/test.json'
    }

    # Load both splits
    dataset = load_dataset('json', data_files=data_files)

    # Define a DatasetReader, with specified column names where input and output are stored.
    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    for ice_num in [5, 10, 20, 30]:
        print(f'\tICE_NUM={ice_num}')
        outpath = f'{args.out_path}/{args.task}/{args.demo}Demo'
        os.makedirs(outpath, exist_ok=True)
        # Define a retriever using the previous `DataLoader`.
        # `ice_num` stands for the number of data in in-context examples.
        if args.demo == 'topk':
            retriever = TopkRetriever(data, ice_num=ice_num)
        if args.demo == 'votek':
            retriever = VotekRetriever(data, ice_num=ice_num)
        if args.demo == 'dpp':
            retriever = DPPRetriever(data, ice_num=ice_num)
        if args.demo == 'mdl':
            retriever = MDLRetriever(data, ice_num=ice_num,
                                     candidate_num=30, select_time=10, seed=1, batch_size=12)
        demo_ids = retriever.retrieve()
        train_ids = retriever.index_ds['id']

        demo_dict = {}
        for i, test_id in enumerate(retriever.test_ds['id']):
            demo = [train_ids[x] for x in demo_ids[i]]
            demo_dict[test_id] = demo
            with open(f'{outpath}/k-{ice_num}.jsonl', 'w+') as f:
                if f.tell() > 0:  # Check if file is not empty
                    f.write('\n')
                json.dump(demo_dict, f)
        del retriever, train_ids, demo_ids, demo_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICL')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--demo', type=str, required=True)

    parser.add_argument('--data_dir', type=str, default='/blue/woodard/share/Relation-Extraction/LLM_feasibility/Data_ICL')
    parser.add_argument('--out_path', type=str, default='/blue/woodard/share/Relation-Extraction/Data')

    args = parser.parse_args()
    main(args)
    print('\tDone.')