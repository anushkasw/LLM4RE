import json
import logging
import os
import random
import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.dataset_readers.prerank_dsr import PrerankDatasetReader
from data_loader import DataProcessor
from config import Config
from transformers import TRANSFORMERS_CACHE

TRANSFORMERS_CACHE = "/blue/woodard/share/HR_Models"

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up global cache dir
os.environ['TRANSFORMERS_CACHE'] = "/blue/woodard/share/HR_Models"

class PreRank:
    def __init__(self, cfg, data_processor) -> None:
        self.cuda_device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")
        self.retriever_model = SentenceTransformer(cfg.retriever_model, cache_folder="/blue/woodard/share/HR_Models").to(self.cuda_device) \
            if cfg.retriever_model != 'none' else None
        self.retriever_model.eval()

        # Initialize rel2id from data_processor if needed
        self.rel2id = data_processor.rel2id

        train_examples = data_processor.get_train_examples()
        test_examples = data_processor.get_test_examples()

        self.dataset_reader = PrerankDatasetReader(
            task=cfg.dataset_reader['task'],
            field=cfg.dataset_reader['field'],
            examples=train_examples,  # Use test_examples for test phase
            tokenizer=self.retriever_model.tokenizer,
        )

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader['dataset_split'] == "train"
        self.method = cfg.method

        if self.method != "random":
            if os.path.exists(cfg.index_file + "_faiss.index") and os.path.exists(cfg.index_file + "_ids.json"):
                logger.info("Loading FAISS index and corresponding IDs from disk...")
                self.all_embeddings = faiss.read_index(cfg.index_file + "_faiss.index")

                # Load the saved IDs
                with open(cfg.index_file + "_ids.json", "r") as f:
                    self.all_ids = json.load(f)
            else:
                self.all_embeddings, self.all_ids = self.create_index(cfg)
    
    def create_index(self, cfg):
        logger.info("Building index with FAISS-GPU...")

        all_embeddings = []
        all_ids = []

        # Forward pass to compute embeddings
        res_list = self.forward(self.dataloader)

        # Collect embeddings and their corresponding IDs
        for res in res_list:
            embed = res['embed']
            if isinstance(embed, np.ndarray):
                embed = torch.tensor(embed)  # Convert NumPy array to PyTorch tensor
            all_embeddings.append(embed.cpu().numpy()) 
            all_ids.append(res['metadata']['id'])

        # Convert embeddings to a NumPy array
        all_embeddings = np.stack(all_embeddings)

        # FAISS index creation
        d = all_embeddings.shape[1]  # Dimension of the embeddings
        index = faiss.IndexFlatIP(d)  # Using cosine similarity
        
        # Move index to GPU
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)  # Move to GPU (use device 0)
        
        # Add embeddings to the FAISS index
        index.add(all_embeddings)

        # Ensure the directory for saving the files exists
        os.makedirs(os.path.dirname(cfg.index_file), exist_ok=True)
        faiss.write_index(faiss.index_gpu_to_cpu(index), cfg.index_file + "_faiss.index")

        # Save the IDs as a JSON file
        with open(cfg.index_file + "_ids.json", "w") as f:
            json.dump(all_ids, f)

        logger.info(f"FAISS index built with size {len(self.dataset_reader)}.")
        return index, all_ids

    def forward(self, dataloader, **kwargs):
        res_list = []
        logger.info(f"Total number of batches: {len(dataloader)}")
        
        # Process batches of data
        for i, entry in enumerate(dataloader):
            with torch.no_grad():
                if i % 500 == 0:
                    logger.info(f"Processing batch {str(i)}")
                    
                metadata = entry.pop("metadata")
                # Tokenize and decode the input IDs back into raw text
                raw_text = self.retriever_model.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True)
                
                # Encode the batch of text without attention_mask
                res = self.retriever_model.encode(
                    raw_text,  # List of raw text sentences
                    show_progress_bar=False, 
                    **kwargs
                )
                
            # Append the embeddings and metadata to the result list
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
            
        return res_list
    
    def knn_search(self, embeddings, num_candidates=1, num_ice=1):
        # Convert the embeddings to NumPy arrays for FAISS
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Search FAISS for nearest neighbors in batch mode
        distances, top_k_indices = self.all_embeddings.search(embeddings, max(num_candidates, num_ice))

        # Convert top_k_indices from NumPy to PyTorch tensors if needed for further operations
        top_k_ids = [[self.all_ids[idx] for idx in top_k_indices[i]] for i in range(top_k_indices.shape[0])]

        if self.is_train:
            top_k_ids = [ids[1:] for ids in top_k_ids]  # Skip the first entry which is the query itself for each batch

        logger.debug(f"Returned indices for in-context examples (num_ice): {[ids[:num_ice] for ids in top_k_ids]}")
        logger.debug(f"Returned indices for candidates (num_candidates): {[ids[:num_candidates] for ids in top_k_ids]}")

        return [ids[:num_ice] for ids in top_k_ids], [ids[:num_candidates] for ids in top_k_ids]

    def search(self, embeddings):
        if self.method == "topk":
            ctxs, candidates = self.knn_search(embeddings, num_candidates=self.num_candidates, num_ice=self.num_ice)
            logger.debug(f"Search returned in-context indices: {ctxs}")
            logger.debug(f"Search returned candidate indices: {candidates}")
            return ctxs, candidates

    def mdl_ranking(self, queries, candidates_list):
        all_candidate_examples_batch = []

        # Fetch the full examples for each candidate ID in the batch
        for candidates in candidates_list:
            full_examples = [self.dataset_reader.dataset_wrapper[candidate_id] for candidate_id in candidates]
            all_candidate_examples_batch.append(full_examples)

        scores = []
        # Evaluate MDL for each query and its corresponding examples
        for query, examples in zip(queries, all_candidate_examples_batch):
            score = self.evaluate_mdl(query, examples)
            scores.append(score)

        # Select the candidate with the lowest MDL score for each query
        selected_indices_batch = [np.argmin(score) for score in scores]

        logger.debug(f"Selected indices batch: {selected_indices_batch}")
        return selected_indices_batch

    def evaluate_mdl(self, query, examples_batch):
        total_mdl_scores_batch = []

        possible_labels = list(self.rel2id.keys())  # rel2id maps relation names to IDs

        for query, examples in zip([query], [examples_batch]):
            total_mdl_score = 0.0

            for label in possible_labels:
                # Construct context for the entire batch
                context_input = self.construct_context([query], [examples], label)
                
                # Compute log probability for the entire batch
                log_probs_batch = self.get_log_probability(context_input, label)
                
                # Multiply log_prob by label probability and sum up
                label_prob = self.get_label_probability(label)
                total_mdl_score += -log_probs_batch.mean().item() * label_prob

            total_mdl_scores_batch.append(total_mdl_score)

        return total_mdl_scores_batch

    def construct_context(self, queries, examples_batch, label):
        # Process each query and corresponding examples in the batch
        contexts = []
        
        for query, examples in zip(queries, examples_batch):
            context = ""
            # Process each example in the current batch
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
            
            # Add the constructed context to the batch
            contexts.append(context)

        return contexts

    def get_log_probability(self, context_inputs, label):
        # Tokenize the input batch with truncation and move to the correct device
        inputs = self.retriever_model.tokenizer(
            context_inputs, 
            return_tensors="pt", 
            max_length=self.retriever_model.tokenizer.model_max_length,
            truncation=True,
            padding=True
        ).to(self.cuda_device)
        
        # Extract the input_ids and attention_mask tensors
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            # Pass both input_ids and attention_mask to the model
            outputs = self.retriever_model({'input_ids': input_ids, 'attention_mask': attention_mask})
            hidden_state = outputs['sentence_embedding']
        
        # Apply log_softmax over the class dimension for the entire batch
        log_probs = torch.log_softmax(hidden_state, dim=-1)
        
        # Handle the indexing based on the actual dimensions
        if hidden_state.dim() == 2:
            # Assuming [batch_size, num_classes]
            log_probs_batch = log_probs[:, self.rel2id[label]]
        elif hidden_state.dim() == 1:
            # Assuming [num_classes]
            log_probs_batch = log_probs[self.rel2id[label]]
        else:
            raise ValueError(f"Unexpected output dimensions: {hidden_state.shape}")
        
        # Return the log probabilities for the batch
        return log_probs_batch

    def get_label_probability(self, label):
        num_labels = len(self.rel2id)
        return 1.0 / num_labels

    def find(self):
        res_list = self.forward(self.dataloader)
        data_list = []
        logger.info("Starting retrieval...")

        # Convert the embeddings from NumPy arrays to PyTorch tensors
        embeddings = torch.stack([torch.tensor(entry['embed']).to(self.cuda_device) for entry in res_list])
        
        ctxs_list, candidates_list = self.search(embeddings)

        # Prepare a batch of queries, ensuring to fetch the full query entry that includes 'token'
        queries = [self.dataset_reader.dataset_wrapper[entry['metadata']['id']] for entry in res_list]

        # Perform batch MDL ranking
        ranked_ctxs_ids_batch = self.mdl_ranking(queries, candidates_list)

        for entry, ranked_ctxs_ids in zip(res_list, ranked_ctxs_ids_batch):
            logger.debug(f"Processing entry: {entry['metadata']}")  # Inspect metadata
            query_full_entry = self.dataset_reader.dataset_wrapper[entry['metadata']['id']]
            logger.debug(f"Query full entry: {query_full_entry}")  # Inspect full entry

            # Assign the ranked context IDs for each query in the batch
            query_full_entry['ctxs'] = ranked_ctxs_ids
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