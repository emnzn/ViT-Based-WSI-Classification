import torch
import torch.nn as nn

class AttentionBasedMIL(nn.Module):
    def __init__(self, attention_func: str):
        super(AttentionBasedMIL, self).__init__()

        self.attention_func = attention_func

        self.attention = nn.Sequential(
            
        )