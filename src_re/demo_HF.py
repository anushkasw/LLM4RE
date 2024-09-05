import torch
import torch.nn.functional as F
from pipelines_HF import HFModelPipelines

class Demo_HF:
    def __init__(self, access_token, model_name, max_tokens, cache_dir):
        self.pipeline = HFModelPipelines(access_token, cache_dir=cache_dir).get_pipeline(model_name)
        self.max_new_tokens = max_tokens
        # self.tokenizer = self.pipeline.tokenizer
        # self.model = self.pipeline.model
        # self.temperature = temperature
        # self.max_tokens = max_tokens
        # self.top_p = top_p
        # self.frequency_penalty = frequency_penalty
        # self.presence_penalty = presence_penalty
        # self.logprobs = logprobs
        # # self.tokenizer.pad_token = self.tokenizer.eos_token
        #
        # # # Add a special pad token if the tokenizer doesn't have one
        # # if self.tokenizer.pad_token is None:
        #     # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Adding a new special [PAD] token
        # #     self.model.resize_token_embeddings(len(self.tokenizer))  # Resizing the model embeddings to accommodate the new token

    def get_multiple_sample(self, messages):
        outputs = self.pipeline(messages, max_new_tokens=self.max_new_tokens, clean_up_tokenization_spaces=True)[0]['generated_text']
        return outputs

        # Tokenize the input prompt
        # inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors='pt')
        # Ensure attention mask is set
        # attention_mask = inputs['attention_mask']
        # Generate output with logits
        # outputs = self.model.generate(
        #     inputs,
        #     # attention_mask=attention_mask,
        #     max_new_tokens=self.max_tokens,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     include_prompt_in_result=False
        # )
        #
        # # Get the generated tokens and scores
        # generated_tokens = outputs.sequences
        # scores = outputs.scores
        #
        # # Decode the generated tokens to text
        # generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        #
        # # Calculate log probabilities
        # logprobs = []
        # for score in scores:
        #     log_prob = F.log_softmax(score, dim=-1)
        #     logprobs.append(log_prob)
        #
        # return [generated_text], logprobs