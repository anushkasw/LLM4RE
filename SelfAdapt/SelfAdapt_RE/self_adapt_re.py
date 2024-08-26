import json
import logging
import os
import random
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.dataset_readers.prerank_dsr import PrerankDatasetReader
from data_loader import DataProcessor
from config import Config

logger = logging.getLogger(__name__)

class PreRank:
    def __init__(self, cfg, data_processor) -> None:
        self.cuda_device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
        self.retriever_model = SentenceTransformer(cfg.retriever_model).to(self.cuda_device) if cfg.retriever_model != 'none' else None
        self.retriever_model.eval()

        # Initialize rel2id from data_processor if needed
        self.rel2id = data_processor.rel2id

        train_examples = data_processor.get_train_examples()
        test_examples = data_processor.get_test_examples()

        self.dataset_reader = PrerankDatasetReader(
            task=cfg.dataset_reader['task'],
            field=cfg.dataset_reader['field'],
            examples=train_examples,  # Use test_examples for test phase
            tokenizer=self.retriever_model.tokenizer
        )

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader['dataset_split'] == "train"
        self.method = cfg.method

        if self.method != "random":
            self.all_embeddings, self.all_ids = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("Building index...")

        all_embeddings = []
        all_ids = []

        # Create the dataloader
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        # Forward pass to compute embeddings
        res_list = self.forward(dataloader)

        # Collect embeddings and their corresponding IDs
        for res in res_list:
            embed = res['embed']
            if isinstance(embed, np.ndarray):
                embed = torch.tensor(embed)  # Convert NumPy array to PyTorch tensor
            all_embeddings.append(embed)
            all_ids.append(res['metadata']['id'])

        # Convert embeddings to a PyTorch tensor
        all_embeddings = torch.stack(all_embeddings)

        # Ensure the directory for saving the files exists
        os.makedirs(os.path.dirname(cfg.index_file), exist_ok=True)

        # Save the embeddings as a PyTorch tensor
        torch.save(all_embeddings, cfg.index_file + "_embeddings.pt")

        # Save the IDs as a JSON file
        with open(cfg.index_file + "_ids.json", "w") as f:
            json.dump(all_ids, f)

        logger.info(f"Index built with size {len(self.dataset_reader)}.")
        return all_embeddings, all_ids

    def forward(self, dataloader, **kwargs):
        res_list = []
        logger.info(f"Total number of batches: {len(dataloader)}")
        for i, entry in enumerate(dataloader):
            with torch.no_grad():
                if i % 500 == 0:
                    logger.info(f"Processing batch {str(i)}")
                metadata = entry.pop("metadata")
                raw_text = self.retriever_model.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True)
                res = self.retriever_model.encode(raw_text, show_progress_bar=False, **kwargs)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    # TopK selection
    def knn_search(self, entry, num_candidates=1, num_ice=1):
        # Convert the embedding to a PyTorch tensor if it is a NumPy array
        embed = entry['embed']
        if isinstance(embed, np.ndarray):
            embed = torch.tensor(embed)
        
        # Add a batch dimension
        embed = embed.unsqueeze(0)

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(embed, self.all_embeddings, dim=1)
        top_k_values, top_k_indices = torch.topk(similarities, max(num_candidates, num_ice))

        if self.is_train:
            top_k_indices = top_k_indices[1:]  # Skip the first entry which is the query itself

        return top_k_indices[:num_ice], top_k_indices[:num_candidates]
    
    def search(self, entry):
        if self.method == "topk":
            return self.knn_search(entry, num_candidates=self.num_candidates, num_ice=self.num_ice)
    
    # MDL ranking
    def mdl_ranking(self, entry, candidates, query):
        all_candidate_idx = [candidates]
        window = len(candidates)

        if window > 1:
            all_candidate_idx = [np.random.permutation(candidates).tolist() for _ in range(window)]

        # Map integer indices to string IDs using self.dataset_reader.keys
        in_context_examples = [
            [self.dataset_reader.dataset_wrapper[self.dataset_reader.keys[i]] for i in candidates_idx]
            for candidates_idx in all_candidate_idx
        ]

        scores = []
        for examples in in_context_examples:
            score = self.evaluate_mdl(query, examples)
            scores.append(score)

        selected_idx = all_candidate_idx[np.argmin(scores)]  # Lower MDL score is better
        return selected_idx

    def evaluate_mdl(self, query, examples):
        total_mdl_score = 0.0

        possible_labels = list(self.rel2id.keys())  # Assuming rel2id maps relation names to IDs

        for label in possible_labels:
            context_input = self.construct_context(query, examples, label)
            log_prob = self.get_log_probability(context_input, label)
            total_mdl_score += -log_prob * self.get_label_probability(label)

        return total_mdl_score

    def construct_context(self, query, examples, label):
        context = ""
        for example in examples:
            if 'token' not in example:
                raise KeyError(f"Expected 'token' key in example, but got {example}")
            # Join the list of tokens into a single string (a sentence)
            token_text = " ".join(example['token'])
            context += f" {token_text} "

        if 'token' not in query:
            raise KeyError(f"Expected 'token' key in query, but got {query}")
        # Join the list of tokens for the query into a single string
        query_token = " ".join(query['token'])
        context += f" {query_token} [LABEL: {label}]"
        
        return context

    def get_log_probability(self, context_input, label):
        # Tokenize the input with truncation and move to the correct device
        inputs = self.retriever_model.tokenizer(
            context_input, 
            return_tensors="pt", 
            max_length=self.retriever_model.tokenizer.model_max_length,  # Use the model's max length
            truncation=True,
            padding=True  # Add padding to ensure consistent sequence length
        ).to(self.cuda_device)
        
        # Extract the input_ids and attention_mask tensors
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            # Pass both input_ids and attention_mask to the model
            outputs = self.retriever_model({'input_ids': input_ids, 'attention_mask': attention_mask})
            
            # DEBUG: Inspect the outputs keys and shape
            # print(f"DEBUG: Outputs keys: {outputs.keys()}")
            
            # Assuming you want to work with 'sentence_embedding' for overall sentence-level tasks
            hidden_state = outputs['sentence_embedding']  # or use 'token_embeddings' if needed
        
        # Apply log_softmax over the class dimension (typically the last dimension)
        log_probs = torch.log_softmax(hidden_state, dim=-1)
        
        # Handle the indexing based on the actual dimensions
        if hidden_state.dim() == 2:
            # Assuming [batch_size, num_classes]
            log_prob = log_probs[0, self.rel2id[label]].item()
        elif hidden_state.dim() == 1:
            # Assuming [num_classes]
            log_prob = log_probs[self.rel2id[label]].item()
        else:
            raise ValueError(f"Unexpected output dimensions: {hidden_state.shape}")
        
        return log_prob

    def get_label_probability(self, label):
        num_labels = len(self.rel2id)
        return 1.0 / num_labels

    def find(self):
        res_list = self.forward(self.dataloader)
        data_list = []
        logger.info("Starting retrieval...")
        for entry in res_list:
            print(f"DEBUG: Entry metadata: {entry['metadata']}")  # Inspect metadata
            query_full_entry = self.dataset_reader.dataset_wrapper[entry['metadata']['id']]
            print(f"DEBUG: Query full entry: {query_full_entry}")  # Inspect full entry

            ctxs, ctxs_candidates = self.search(entry)
            ranked_ctxs = self.mdl_ranking(query_full_entry, ctxs_candidates, query_full_entry)

            query_full_entry['ctxs'] = ranked_ctxs
            data_list.append(query_full_entry)

        logger.info("Saving output...")
        with open(self.output_file, "w") as f:
            json.dump(data_list, f)

def main():
    config = Config()

    config.output_file = "/blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfAdapt/SelfAdapt_RE/output/test.json"
    config.index_file = "/blue/woodard/share/Relation-Extraction/bell/LLM4RE/SelfAdapt/SelfAdapt_RE/index/"

    logger.info(config.__dict__)
    if not config.overwrite and os.path.exists(config.output_file):
        logger.info(f'{config.output_file} already exists, skipping')
        return
    
    data_processor = DataProcessor(config)

    dense_retriever = PreRank(config, data_processor)
    random.seed(config.rand_seed)
    np.random.seed(config.rand_seed)
    dense_retriever.find()

if __name__ == "__main__":
    main()