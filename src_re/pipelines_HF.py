# Import packages
import os
import time
import transformers
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from pathlib import Path
from datetime import timedelta

# Enable progress bar
transformers.logging.set_verbosity_info()

# Define the universal model pipeline class
class HFModelPipelines:
    '''
    This pipeline should be able to access all available HF models in transformers.
    model_id: The official name of the model (please check HF).
    '''
    def __init__(self, access_token, cache_dir=None):
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.access_token = access_token
        os.environ['HF_AUTH_TOKEN'] = access_token
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir  # Set the cache directory environment variable
        os.environ['HF_HOME'] = self.cache_dir  # Set the home directory for Hugging Face
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Setting for gpu
        os.environ['HUGGINGFACE_TOKEN'] = f"{self.cache_dir}/token" # Set path for the token
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True) # Check if the path is exist
        login(token=access_token, add_to_git_credential=True)

    def create_pipeline(self, model_id):
        print(f'Loading {model_id}')
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        print('Loading model')
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, cache_dir=self.cache_dir)
        model.generation_config.pad_token_id = tokenizer.eos_token_id

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            return_full_text=False,
        )

    def get_pipeline(self, model_name):
        return self.create_pipeline(model_name)
