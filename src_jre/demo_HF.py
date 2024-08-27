import torch
import torch.nn.functional as F
from pipelines_HF import HFModelPipelines

class Demo_HF:
    def __init__(self, access_token, model_name, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, logprobs, cache_dir):
        self.pipeline = HFModelPipelines(access_token, cache_dir=cache_dir).get_pipeline(model_name)
        self.tokenizer = self.pipeline.tokenizer
        self.model = self.pipeline.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs

    def get_multiple_sample(self, prompt):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.pipeline.device)

        # Generate output with logits
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=self.max_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )

        # Get the generated tokens and scores
        generated_tokens = outputs.sequences
        scores = outputs.scores

        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Calculate log probabilities
        logprobs = []
        for score in scores:
            log_prob = F.log_softmax(score, dim=-1)
            logprobs.append(log_prob)

        return [generated_text], logprobs