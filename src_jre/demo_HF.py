import torch
import torch.nn.functional as F
from pipelines_HF import HFModelPipelines

class Demo_HF:
    def __init__(self, access_token, model_name, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, logprobs, cache_dir):
        self.pipeline = HFModelPipelines(access_token, cache_dir=cache_dir).get_pipeline(model_name)
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = self.pipeline.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs

    def get_multiple_sample(self, prompt):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.pipeline.device)

        # Ensure attention mask is set
        attention_mask = inputs['attention_mask']

        # Generate output with logits
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id
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