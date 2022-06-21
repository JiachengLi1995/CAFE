import os
import math
import numpy as np

import torch
from torch import nn as nn


class POPModel(nn.Module):
    def __init__(self, args, pretrained_item_vectors=None):
        super().__init__()
        self.item_freq = torch.Tensor(args.item_freq).unsqueeze(dim=0).cuda()

    def forward(self, 
                tokens, 
                meta_tokens=None, 
                candidates=None, 
                meta_candidates=None, 
                length=None, mode="train",
                users=None,):
        if mode == "train":
            return torch.tensor(0.0)

        logits = self.item_freq.repeat(tokens.size(0), 1)
        if candidates is None:
            return logits, logits
        else:
            logits = logits[torch.arange(logits.size(0), device=logits.device).unsqueeze(-1), candidates]
            return logits, logits

    def to_device(self, device):
        return self.to(device)
        
    def device_state_dict(self):
        return self.state_dict()