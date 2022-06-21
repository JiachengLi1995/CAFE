from .base import AbstractTrainer
from .utils import SampleRanker, Ranker

class SASRecSampleTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only=False):
        super().__init__(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only)

        if args.eval_all:
            self.ranker = Ranker(self.metric_ks, user2seq)
        else:
            self.ranker = SampleRanker(self.metric_ks)
        
        #self.ranker_meta = SampleRanker(self.metric_ks)

    @classmethod
    def code(cls):
        return 'sasrec_sample'

    def calculate_loss(self, batch):
        users, tokens, meta_tokens, candidates, meta_candidates = batch

        param_dict = {'users': users, 'tokens': tokens, 'meta_tokens': meta_tokens, 'candidates': candidates,
                      'meta_candidates': meta_candidates, 'mode': 'train'}

        loss = self.model(**param_dict)  # scores, loss

        return loss

    def calculate_metrics(self, batch, mode):
        users, seqs, meta_seqs, candidates, meta_candidates, length, labels, meta_labels = batch

        if self.args.eval_all:
            candidates = None
            meta_candidates = None

        param_dict = {'users': users, 'tokens': seqs, 'meta_tokens': meta_seqs, 'candidates': candidates, 
                      'length': length, 'mode': mode, 'meta_candidates': meta_candidates}

        scores, scores_meta = self.model(**param_dict)  # B x T x C

        if self.args.eval_all:
            res = self.ranker(scores, labels, users=users)
        else:
            res = self.ranker(scores)
        
        #res_meta = self.ranker_meta(scores_meta)

        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]

            # metrics["NDCG_meta@%d" % k] = res_meta[2*i]
            # metrics["Recall_meta@%d" % k] = res_meta[2*i+1]
            
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        return metrics
