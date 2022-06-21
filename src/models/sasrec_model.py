import numpy as np

import torch
from torch import nn as nn

from src.utils.utils import fix_random_seed_as


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

class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_layernorms = torch.nn.ModuleList()
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

    def forward(self, seqs, timeline_mask):

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

class Embedding(nn.Module):
    def __init__(self, 
                 num_items, 
                 args, 
                 pad_token, 
                 pretrained_item_vectors=None, 
                 use_pretrained_vectors=False
                 ):
        super().__init__()

        self.args = args
        self.use_pretrained_vectors = use_pretrained_vectors
        self.num_items = num_items

        self.item_emb = torch.nn.Embedding(self.num_items+1, args.trm_hidden_dim, padding_idx=pad_token)
        if self.use_pretrained_vectors and type(pretrained_item_vectors)==np.ndarray:
            assert pretrained_item_vectors.shape[0] == self.num_items
            self.item_emb.weight[:self.num_items].data.copy_(torch.from_numpy(pretrained_item_vectors))
            #self.item_emb.weight.requires_grad = False
            self.target_emb = torch.nn.Embedding.from_pretrained(self.item_emb.weight, freeze=True)
            print('Using pretrained product vectors! Loading pretrained product vectors.')
        elif self.use_pretrained_vectors and pretrained_item_vectors == None:
            self.target_emb = torch.nn.Embedding(self.num_items+1, args.trm_hidden_dim, padding_idx=pad_token)
            print('Using pretrained product vectors! Vector is None. This is expected when test.')
        else:
            print('Training without pretrained product vectors!')

        self.pad_token = pad_token
        self.pos_emb = torch.nn.Embedding(args.trm_max_len, args.trm_hidden_dim)
        self.emb_dropout = torch.nn.Dropout(p=args.trm_dropout)

    def forward(self, log_seqs):
        seqs = self.lookup_input(log_seqs)
        seqs *= self.args.trm_hidden_dim ** 0.5
        positions = torch.arange(log_seqs.shape[1]).long().unsqueeze(0).repeat([log_seqs.shape[0], 1])
        seqs = seqs + self.pos_emb(positions.to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == self.pad_token).bool()
        return seqs, timeline_mask

    def lookup_input(self, x):
        return self.item_emb(x)

    def lookup_target(self, x):
        if self.use_pretrained_vectors:
            return self.target_emb(x)
        else:
            return self.item_emb(x)

    def all_predict(self, log_feats):
        if self.use_pretrained_vectors:
            w = self.target_emb.weight.transpose(1, 0)
        else:
            w = self.item_emb.weight.transpose(1, 0)
        return torch.matmul(log_feats, w)



class SASRecModel(nn.Module):
    def __init__(self, args, pretrained_item_vectors=None):
        super().__init__()
        self.args = args
        fix_random_seed_as(args.model_init_seed)

        self.loss = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.num_items = args.num_items
        self.num_meta = args.num_meta
        self.pad_token = args.pad_token
        self.product_embedding = Embedding(self.num_items, args, args.pad_token, pretrained_item_vectors, args.use_pretrained_vectors)
        self.product_attention = SelfAttention(args)
        self.use_pretrained_vectors = args.use_pretrained_vectors
                
    def forward(self, 
                tokens, 
                meta_tokens=None, 
                candidates=None, 
                meta_candidates=None, 
                length=None, mode="train",
                users=None
                ):

        idx1, idx2 = self.select_predict_index(tokens) if mode == "train" else (torch.arange(tokens.size(0)), length.squeeze())
        feature, time_mask = self.product_embedding(tokens)
 
        feature, attn_weights = self.product_attention(feature, time_mask)
        
        feature = feature[idx1, idx2]
        
        if mode == "train":
            candidates = candidates[idx1, idx2]

            pos_seqs = candidates[:, 0]
            neg_seqs = candidates[:, -1]


            pos_embs = self.product_embedding.lookup_target(pos_seqs)
            neg_embs = self.product_embedding.lookup_target(neg_seqs)

            pos_logits = (feature * pos_embs).sum(dim=-1)
            neg_logits = (feature * neg_embs).sum(dim=-1)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=neg_logits.device)
            loss = self.loss(pos_logits, pos_labels)
            loss += self.loss(neg_logits, neg_labels)

            return loss

        else:
            if candidates is not None:
                log_feats = feature.unsqueeze(1) # x is (batch_size, 1, embed_size)
                w = self.product_embedding.lookup_target(candidates).transpose(2,1) # (batch_size, embed_size, candidates)
                logits = torch.bmm(log_feats, w).squeeze(1) # (batch_size, candidates)
            else:
                logits = self.product_embedding.all_predict(feature)
            
            return logits, logits
        

    def select_predict_index(self, x):
        return (x!=self.pad_token).nonzero(as_tuple=True)            

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n):
                    try:
                        torch.nn.init.xavier_uniform_(p.data)
                    except:
                        pass # just ignore those failed init layers

    

    def to_device(self, device):
        return self.to(device)
        
    def device_state_dict(self):
        return self.state_dict()