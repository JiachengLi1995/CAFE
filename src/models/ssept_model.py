import os
import math
import numpy as np
import random

import torch
from torch import nn as nn
import torch.backends.cudnn as cudnn

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SSEPTModel(nn.Module):
    def __init__(self, args, pretrained_item_vectors=None):
        super().__init__()
        self.args = args
        fix_random_seed_as(args.model_init_seed)

        # parameters
        self.item_num = args.num_items
        self.user_num = args.num_users
        self.pad_token = args.pad_token
        self.item_hidden_dim = args.trm_hidden_dim // 2
        self.user_hidden_dim = args.trm_hidden_dim // 2
        args.trm_hidden_dim = self.item_hidden_dim + self.user_hidden_dim
        self.loss = nn.BCELoss()

        # embedding for SASRec, sum of positional, token embeddings
        self.item_emb = torch.nn.Embedding(self.item_num+2, self.item_hidden_dim, padding_idx=self.pad_token)
        self.user_emb = torch.nn.Embedding(self.user_num+2, self.user_hidden_dim, padding_idx=-1)
        self.pos_emb = torch.nn.Embedding(args.trm_max_len, args.trm_hidden_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.trm_dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)

        for _ in range(args.trm_num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.trm_hidden_dim,
                                                            args.trm_num_heads,
                                                            args.trm_dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.trm_hidden_dim, args.trm_dropout)
            self.forward_layers.append(new_fwd_layer)

        # weights initialization
        self.init_weights()

    def scale(self, x, hidden):
        return x * (hidden ** 0.5)

    def log2feats(self, log_seqs, users):
        seqs_item = self.scale(self.item_emb(log_seqs), self.item_emb.embedding_dim)
        seqs_user = self.scale(self.user_emb(users), self.user_emb.embedding_dim).repeat(1, log_seqs.size(1), 1)

        positions = torch.arange(log_seqs.shape[1]).long().unsqueeze(0).repeat([log_seqs.shape[0], 1])
        seqs = torch.cat([seqs_item, seqs_user], dim=-1) + self.pos_emb(positions.to(seqs_item.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, seqs_user

    def forward(self, tokens, users, candidates=None, length=None, mode="train", meta_tokens=None, meta_candidates=None):

        # embedding and mask creation
        idx1, idx2 = self.select_predict_index(tokens) if mode == "train" else (torch.arange(tokens.size(0)), length.squeeze())
        tokens, u = self.log2feats(tokens, users)

        # select valid x
        tokens = tokens[idx1, idx2]
        u = u[idx1, idx2]

        # similarity calculation
        if mode != "train":
            logits = self.test_similarity_score(tokens, candidates)
            return logits, logits
        else:
            candidates = candidates[idx1, idx2]
            logits = self.train_similarity_score(tokens, u, candidates)
            scores = logits.sigmoid()
            labels = torch.zeros_like(logits, device=logits.device)
            labels[:, 0] = 1
            loss = self.loss(scores, labels)
            
            return loss


    def select_predict_index(self, x):
        return (x!=self.pad_token).nonzero(as_tuple=True)

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            # compute bounds with CDF
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            # sample uniformly from [2l-1, 2u-1] and map to normal 
            # distribution with the inverse error function
            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n):
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def train_similarity_score(self, x, u, candidates):
        x = x.unsqueeze(1) # x is (batch_size, 1, embed_size)
        w = self.scale(self.item_emb(candidates), self.item_hidden_dim) # (batch_size, candidates, item_embed_size)
        w = torch.cat([w, u.unsqueeze(1).repeat(1,w.size(1),1)], dim=-1) # (batch_size, candidates, embed_size)
        w = w.transpose(2,1) # (batch_size, candidates, embed_size)
        return torch.bmm(x, w).squeeze(1) # (batch_size, candidates)

    def test_similarity_score(self, x, candidates):
        if candidates is None:
            # item part 
            w = self.scale(self.item_emb.weight, self.item_hidden_dim).transpose(1,0) # (item_emb, item_num)
            x = x[:, :w.size(0)] # (batch, embed_size) -> (batch, item_embed_size)
            # user part doesn't change the order, so omit it in testing
            return torch.matmul(x, w)
        else:
            x = x.unsqueeze(1) # x is (batch_size, 1, embed_size)
            w = self.scale(self.item_emb(candidates), self.item_hidden_dim).transpose(2,1) 
            x = x[:, :, :w.size(1)] # (batch, 1, embed_size) -> (batch, 1, item_embed_size)
            return torch.bmm(x, w).squeeze(1) # (batch_size, candidates)

    def to_device(self, device):
        return self.to(device)
        
    def device_state_dict(self):
        return self.state_dict()

