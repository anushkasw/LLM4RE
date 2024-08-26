import torch
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper

class PrerankDatasetReader(torch.utils.data.Dataset):
    def __init__(self, task, examples=None, dataset_path=None, dataset_split=None,
                 ds_size=None, tokenizer=None, field="token") -> None:
        self.tokenizer = tokenizer
        self.field = field

        if examples is not None:
            # Store data as a dictionary and keep a list of keys
            self.dataset_wrapper = {entry['id']: entry for entry in examples}
            self.keys = list(self.dataset_wrapper.keys())
        else:
            data_list = get_dataset_wrapper(task)(dataset_path=dataset_path,
                                                  dataset_split=dataset_split,
                                                  ds_size=ds_size)
            self.dataset_wrapper = {entry['id']: entry for entry in data_list}
            self.keys = list(self.dataset_wrapper.keys())

    def __getitem__(self, index):
        # Map the integer index to the string ID
        key = self.keys[index]
        entry = self.dataset_wrapper[key]
        enc_text = entry[self.field]

        if isinstance(enc_text, list):
            enc_text = " ".join(enc_text)

        if not isinstance(enc_text, str):
            raise ValueError(f"Expected a string for field '{self.field}', but got {type(enc_text)}")

        tokenized_inputs = self.tokenizer.encode_plus(
            enc_text, truncation=True, return_tensors='pt', padding='longest'
        )

        return {
            'input_ids': tokenized_inputs.input_ids.squeeze(),
            'attention_mask': tokenized_inputs.attention_mask.squeeze(),
            "metadata": {"id": key}
        }

    def __len__(self):
        return len(self.keys)