import numpy as np

import torch
from torch import nn as nn

from src.utils.utils import fix_random_seed_as


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
        

class TRMBlock(nn.Module):
    def __init__(self, args, pad_token):
        super().__init__()

        self.args = args
        self.pad_token = pad_token

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)

        for _ in range(args.trm_num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(args.trm_hidden_dim,
                                                            args.trm_num_heads,
                                                            args.trm_dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.trm_hidden_dim, args.trm_dropout)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs, seqs):

        timeline_mask = (log_seqs == self.pad_token).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        attn_output_weights = []

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_output_weight = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            
            attn_output_weights.append(attn_output_weight)
            
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, attn_output_weights

class TimeAwareMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = nn.Linear(hidden_size, hidden_size)
        self.K_w = nn.Linear(hidden_size, hidden_size)
        self.V_w = nn.Linear(hidden_size, hidden_size)

        self.time_w = nn.Linear(hidden_size, hidden_size)
        self.time_act = nn.ReLU()
        self.time_proj = nn.Linear(hidden_size, 1)
        self.time_sig = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, values, time_mask, attn_mask, time_matrix_K):

        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(values)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))        
        time_weights = self.time_sig(self.time_proj(self.time_act(self.time_w(time_matrix_K))).squeeze())
        attn_weights += torch.log(time_weights)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(queries.device)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs, attn_weights

class TimeTRMBlock(nn.Module):
    def __init__(self, args, pad_token):
        super().__init__()

        self.args = args
        self.pad_token = pad_token

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.time_pos_proj = nn.Linear(args.trm_hidden_dim * 50, args.trm_hidden_dim)

        self.last_layernorm = nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)

        for _ in range(args.trm_num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  TimeAwareMultiHeadAttention(args.trm_hidden_dim,
                                                            args.trm_num_heads,
                                                            args.trm_dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.trm_hidden_dim, args.trm_dropout)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs, seqs, time_matrix_K):

        timeline_mask = (log_seqs == self.pad_token).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        attn_output_weights = []

        for i in range(len(self.attention_layers)):

            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_output_weight = self.attention_layers[i](Q, seqs, seqs, 
                                            timeline_mask, attention_mask,
                                            time_matrix_K)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            
            attn_output_weights.append(attn_output_weight)
            
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, attn_output_weights