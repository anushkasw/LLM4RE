class Config:
    def __init__(self):
        # General settings
        self.output_file = None  # Path to save the output results
<<<<<<< HEAD
        self.rand_seed = 42
        self.num_candidates = 10  # Number of candidates to retrieve in TopK search
        self.num_ice = 20  # Number of in-context examples to select
=======
        self.rand_seed = 1
        self.num_candidates = 10  # Number of candidates to retrieve in TopK search
        self.num_ice = 8  # Number of in-context examples to select
>>>>>>> 0cc934febc0090f3eb22c2be6891852d9004b927
        self.batch_size = 64
        self.cuda_device = "cuda:0"  # CUDA device for processing
        self.overwrite = True  # Whether to overwrite existing output files
        self.method = "topk"  # Method for candidate selection, "topk" in this case

        # Model settings
        self.model_name = "gpt2-xl"  # Name of the model for MDL evaluation
        self.retriever_model = "all-mpnet-base-v2"  # Sentence transformer model for TopK retrieval

        # Dataset reader settings
        self.dataset_reader = {
            "task": "dummy_tacred",  # Task name or dataset identifier
            "model_name": self.model_name,
            "field": "token",  # Field used for embeddings or input text
            "dataset_split": "validation",  # Dataset split (train/validation/test)
            "dataset_path": "/blue/woodard/share/Relation-Extraction/Data"  # Path to the dataset, set this as needed
        }

        # Index reader settings
        self.index_reader = {
            "task": "dummy_tacred",  # Task name or dataset identifier for indexing
            "model_name": self.model_name,
            "field": "token",  # Embedding field
            "dataset_split": "train",  # Dataset split used for indexing
            "dataset_path": "/blue/woodard/share/Relation-Extraction/Data"  # Path to the index dataset, set this as needed
        }

        # Model settings for MDL ranking
        self.model = {
            "pretrained_model_name_or_path": self.model_name,  # Pretrained model for MDL scoring
        }

        # Add data_dir attribute
        self.data_dir = "/blue/woodard/share/Relation-Extraction/Data"  # Base directory for datasets
        self.task = "dummy_tacred"
        self.demo = "sel_adapt"

# Instantiate the configuration
config = Config()