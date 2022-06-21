import math
import torch
from torch import nn as nn
import torch.nn.functional as F

MAX_VAL = 1e4

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, activate="gelu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activate=="gelu" else nn.ReLU()

    def forward(self, x):
        return self.w_2(self.activation(self.dropout(self.w_1(x))))

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout, residual=True):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, sublayer):
        if self.residual:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(self.dropout(sublayer(x)))


class Attention(nn.Module):
    def __init__(self, max_len, head_size, args=None):
        super().__init__()

        self.args = args
        
        self.global_num = args.trm_num_heads
        self.local_num = args.local_num_heads

        if self.local_num > 0:

            self.abs_pos_emb_key = nn.Embedding(max_len, head_size * self.local_num) 
            self.abs_pos_emb_query = nn.Embedding(max_len, head_size * self.local_num) 
            self.rel_pos_emb = nn.Embedding(2 * max_len - 1, head_size * self.local_num)
            position_ids_l = torch.arange(max_len, dtype=torch.long).view(-1, 1)
            position_ids_r = torch.arange(max_len, dtype=torch.long).view(1, -1)
            self.distance = position_ids_r - position_ids_l + max_len - 1
            self.mlps = nn.ModuleList([nn.Linear(head_size, 1) for _ in range(self.local_num)])
            self.sigmoid = nn.Sigmoid()

    "Compute 'Scaled Dot Product Attention"
    def forward(self, query, key, value, mask=None, dropout=None):
        b, h, l, d_k = query.size()
        query_g, key_g, value_g = query[:, :self.global_num, ...], key[:, :self.global_num, ...], value[:, :self.global_num, ...]

        scores_g = torch.matmul(query_g, key_g.transpose(-2, -1)) / math.sqrt(query_g.size(-1))
        scores_g = scores_g.masked_fill(mask, -MAX_VAL)
        p_attn_g = dropout(F.softmax(scores_g, dim=-1))
        value_g = torch.matmul(p_attn_g, value_g)


        if self.local_num > 0:
            query_l, key_l, value_l = query[:, self.global_num:, ...], key[:, self.global_num:, ...], value[:, self.global_num:, ...]
            index = self.distance.to(query_l.device)[-1]
            query_l = query_l + self.abs_pos_emb_query(index).unsqueeze(0).unsqueeze(0).view(1,-1,l,d_k)
            key_l = key_l + self.abs_pos_emb_key(index).unsqueeze(0).unsqueeze(0).view(1,-1,l,d_k)

            scores_l = torch.matmul(query_l, key_l.transpose(-2, -1)) / math.sqrt(query_l.size(-1))

            rel_pos_embedding = self.rel_pos_emb(self.distance.to(scores_l.device)).view(l, -1, self.local_num, d_k).permute(2,0,1,3).unsqueeze(0)
            inputs = rel_pos_embedding.repeat(b,1,1,1,1) + value_l.unsqueeze(dim=-2) + value_l.unsqueeze(dim=-3)# + self.user_proj(users).view(b, l, -1, d_k).permute(0,2,1,3).unsqueeze(-2)

            reweight = torch.cat([self.mlps[i](inputs[:, i, ...]).squeeze(-1).unsqueeze(1) for i in range(self.local_num)], dim=1)
            scores_l = scores_l + reweight

            scores_l = scores_l.masked_fill(mask, -MAX_VAL)
            p_attn_l = dropout(F.softmax(scores_l, dim=-1))
            value_l = torch.matmul(p_attn_l, value_l)



            if self.global_num > 0:
                return torch.cat([value_g, value_l], dim=1), (p_attn_g, p_attn_l)
            else:
                return value_l, p_attn_l


        else:

            return value_g, p_attn_g


class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, args=None):
        super().__init__()

        hidden_dim = args.trm_hidden_dim
        num_heads = args.trm_num_heads + args.local_num_heads
        assert hidden_dim % num_heads == 0
        dropout = args.trm_att_dropout
        max_len = args.trm_max_len

        self.head_size = hidden_dim // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.attention = Attention(max_len=max_len, head_size=self.head_size, args=args)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        '''
        query, key, value: [batch, seq_len, hidden]
        '''

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)

        return self.output_linear(x)

class LockerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadedAttention(args=args)
        self.feed_forward = PositionwiseFeedForward(d_model=args.trm_hidden_dim, d_ff=args.trm_hidden_dim, dropout=args.trm_dropout)

        self.input_sublayer = SublayerConnection(size=args.trm_hidden_dim, dropout=args.trm_dropout)
        self.output_sublayer = SublayerConnection(size=args.trm_hidden_dim, dropout=args.trm_dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x

class Locker(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.transformer_blocks = nn.ModuleList([LockerBlock(args=args) for _ in range(args.trm_num_blocks)])

    def forward(self, seqs, timeline_mask):

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        attn_output_weights = [] # useless but consistent with SASRec

        for transformer in self.transformer_blocks:
            seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
            seqs = transformer(seqs, attention_mask)


        return seqs, attn_output_weights


