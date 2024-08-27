# Import packages
import os
import time
import transformers
import torch
# import logging

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
        print('Loading model')
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True, cache_dir=self.cache_dir, low_cpu_mem_usage=True)

        # # Parallel processing - CUDA
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     model = torch.nn.DataParallel(model)
        # model.to(self.device)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    def get_pipeline(self, model_name):
        return self.create_pipeline(model_name)
