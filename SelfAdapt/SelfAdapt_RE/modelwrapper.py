import torch.nn as nn

class MyModelWrapper(nn.Module):
    def __init__(self, model):
        super(MyModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        # If your model only expects input_ids, you might ignore attention_mask
        return self.model(input_ids)  # Modify this based on your model's needs