import json
import logging
from collections import defaultdict
import faiss
import hydra
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.dataset_readers.prerank_dsr import PrerankDatasetReader

logger = logging.getLogger(__name__)

class PreRank:
    def __init__(self, cfg) -> None:
        self.cuda_device = cfg.cuda_device

        # Load the SentenceTransformer model
        self.retriever_model = SentenceTransformer(cfg.retriever_model).to(self.cuda_device) if cfg.retriever_model != 'none' else None
        self.retriever_model.eval()

        # Prepare the dataset
        self.dataset_reader = PrerankDatasetReader(task_name=cfg.dataset_reader.task_name,
                                                   field=cfg.dataset_reader.field,
                                                   dataset_path=cfg.dataset_reader.dataset_path,
                                                   dataset_split=cfg.dataset_reader.dataset_split,
                                                   tokenizer=self.retriever_model.tokenizer)

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"
        self.method = cfg.method

        self.index_reader = PrerankDatasetReader(task_name=cfg.index_reader.task_name,
                                                 field=cfg.index_reader.field,
                                                 dataset_path=cfg.index_reader.dataset_path,
                                                 dataset_split=cfg.index_reader.dataset_split,
                                                 tokenizer=self.retriever_model.tokenizer)

        # Create an index if the method isn't random
        if self.method != "random":
            self.index = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("Building index...")
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(self.index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(768)))
        res_list = self.forward(dataloader)

        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        cpu_index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(cpu_index, cfg.index_file)
        logger.info(f"Index built with size {len(self.index_reader)}.")
        return index

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

    def knn_search(self, entry, num_candidates=1, num_ice=1):
        embed = np.expand_dims(entry['embed'], axis=0)
        near_ids = self.index.search(embed, max(num_candidates, num_ice) + 1)[1][0].tolist()
        near_ids = near_ids[1:] if self.is_train else near_ids
        return near_ids[:num_ice], [[i] for i in near_ids[:num_candidates]]

    def search(self, entry):
        if self.method == "topk":
            return self.knn_search(entry, num_candidates=self.num_candidates, num_ice=self.num_ice)

    def find(self):
        res_list = self.forward(self.dataloader)
        data_list = []
        logger.info("Starting retrieval...")
        for entry in res_list:
            data = self.dataset_reader.dataset_wrapper[entry['metadata']['id']]
            ctxs, ctxs_candidates = self.search(entry)
            data['ctxs'] = ctxs
            data['ctxs_candidates'] = ctxs_candidates
            data_list.append(data)

        logger.info("Saving output...")
        with open(self.output_file, "w") as f:
            json.dump(data_list, f)


@hydra.main(config_path="configs", config_name="prerank")
def main(cfg):
    logger.info(cfg)
    if not cfg.overwrite:
        if os.path.exists(cfg.output_file):
            logger.info(f'{cfg.output_file} already exists, skipping')
            return
    dense_retriever = PreRank(cfg)
    random.seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)
    dense_retriever.find()

if __name__ == "__main__":
    main()