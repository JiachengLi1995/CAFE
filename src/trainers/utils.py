import torch
import torch.nn as nn

MAX_VAL = 1e4
THRESHOLD = 0.5

class SampleRanker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks

    def forward(self, scores):
        predicts = scores[:, 0].unsqueeze(-1) # gather perdicted values
        valid_length = scores.size()[-1] - 1
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC
        return res + [0]

class Ranker(nn.Module):
    def __init__(self, metrics_ks, user2seq):
        super().__init__()
        self.ks = metrics_ks
        self.user2seq = user2seq

    def forward(self, scores, labels, seqs=None, users=None):
        labels = labels.squeeze(-1)
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        if seqs is not None:
            scores[torch.arange(scores.size(0)).unsqueeze(-1), seqs] = -MAX_VAL # mask the rated items
        if users is not None:
            for i in range(len(users)):
                scores[i][self.user2seq[users[i].item()]] = -MAX_VAL
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC
        return res + [0]
