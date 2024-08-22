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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing") 
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

    data = []
    all_sentences = []
    all_labels = []

    for step, rel_id in enumerate(examples):
        for sample in examples[rel_id]:
            tokens = sample['token']  
            sentence = ' '.join(tokens)
            sentence = sentence.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ``", "``").replace("'' ", "''")

            original_input = args.prefix + sentence + args.infix + args.postfix
            label = rel_id
            label_token = data_processor.rel2prompt[sample['relation']]

            label_dependent_input = original_input.replace(args.label_token, label_token)

            # Accumulate sentences for batch processing
            all_sentences.append(label_dependent_input)
            all_labels.append({
                "index": step,
                "label": label,
                "sentence": sentence
            })

            # Process batch when batch size is reached
            if len(all_sentences) >= args.batch_size:
                # Generate in-context samples using the pipeline in batch
                generated_outputs = text_generation_pipeline(
                    all_sentences,
                    max_length=max(len(s) + args.generation_max_length for s in all_sentences),
                    min_length=args.generation_min_length,
                    temperature=args.temperature,
                    truncation=True,
                    do_sample=True,
                    pad_token_id=text_generation_pipeline.tokenizer.eos_token_id
                )

                for i, output in enumerate(generated_outputs):
                    generated_text = output[0] if isinstance(output, list) else output['generated_text'].strip()
                    
                    row_dict = {
                        "index": all_labels[i]["index"],
                        "label": all_labels[i]["label"],
                        "sentence": all_labels[i]["sentence"],
                        "generated_samples": generated_text
                    }
                    data.append(row_dict)

                # Clear the lists after processing the batch
                all_sentences = []
                all_labels = []

    # Process any remaining sentences that didn't fill up the last batch
    if all_sentences:
        generated_outputs = text_generation_pipeline(
            all_sentences,
            max_length=max(len(s) + args.generation_max_length for s in all_sentences),
            min_length=args.generation_min_length,
            temperature=args.temperature,
            truncation=True,
            do_sample=True,
            pad_token_id=text_generation_pipeline.tokenizer.eos_token_id
        )

        for i, output in enumerate(generated_outputs):
            row_dict = {
                "index": all_labels[i]["index"],
                "label": all_labels[i]["label"],
                "sentence": all_labels[i]["sentence"],
                "generated_samples": output['generated_text'].strip()
            }
            data.append(row_dict)

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