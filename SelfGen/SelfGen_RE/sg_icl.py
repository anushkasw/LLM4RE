import random
import os
import logging
import argparse
import json

from transformers import set_seed
from data_loader import DataProcessor
from pipelines_HF import HFModelPipelines

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate in-context samples for self-generated in-context learning")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", type=str, default='', required=False)
    parser.add_argument("--infix", type=str, default='', required=False)
    parser.add_argument("--postfix", type=str, default='', required=False)
    parser.add_argument("--label_token", type=str, default="[LABEL]")
    parser.add_argument("--generation_max_length", type=int, default=10)
    parser.add_argument('--generation_min_length', type=int, default=10)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--access_token", type=str, required=True, help="Hugging Face access token") 
    parser.add_argument("--demo", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    generation_writer = os.path.join(args.output_dir, "generated_samples.json")
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    data_processor = DataProcessor(args)
    examples = data_processor.get_test_examples()  # or get_train_examples()

    # Initialize the HFModelPipelines class and get the pipeline
    hf_pipeline = HFModelPipelines(access_token=args.access_token)
    text_generation_pipeline = hf_pipeline.get_pipeline(args.model_name_or_path)

    logger.info("***** Generating In-Context Samples *****")

    text_generation_pipeline.model.eval()
    
    with open(generation_writer, 'w') as file_writer: 
        data = []

        for step, rel_id in enumerate(examples):
            for sample in examples[rel_id]:
                tokens = sample['token']  
                sentence = ' '.join(tokens)
                sentence = sentence.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ``", "``").replace("'' ", "''")
                
                original_input = args.prefix + sentence + args.infix + args.postfix
                label = rel_id
                label_token = data_processor.rel2prompt[sample['relation']]

                label_dependent_input = original_input.replace(args.label_token, label_token)

                # Generate in-context samples using the pipeline
                generated_outputs = text_generation_pipeline(
                    label_dependent_input,
                    max_length=len(label_dependent_input) + args.generation_max_length,
                    min_length=len(label_dependent_input) + args.generation_min_length,
                    temperature=args.temperature,
                    truncation=True,
                    do_sample=True,
                    pad_token_id=text_generation_pipeline.tokenizer.eos_token_id
                )

                # Process the generated outputs
                generated_texts = [output['generated_text'].strip() for output in generated_outputs]

                # Create a dictionary for each row
                row_dict = {
                    "index": step,
                    "label": label,
                    "sentence": sentence,
                    "generated_samples": generated_texts
                }

                # Append the dictionary to the list
                data.append(row_dict)

        # # Write the entire list of dictionaries to a JSON file
        # json.dump(data, file_writer, indent=4)
    # Ensure data is being written correctly
    if data:
        logger.info(f"Writing {len(data)} samples to the JSON file.")
        with open(generation_writer, 'w') as file_writer:
            json.dump(data, file_writer, indent=4)
    else:
        logger.error("No data was generated to write to the JSON file.")

    logger.info("In-Context Sample Generation Complete.")

if __name__ == "__main__":
    main()