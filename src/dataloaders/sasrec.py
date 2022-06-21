import random
import torch
import torch.utils.data as data_utils
import numpy as np
from copy import deepcopy

class SASRecDataloader(object):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)

        self.train = dataset.train
        self.val = dataset.val
        self.test = dataset.test
        self.smap = dataset.smap

        self.train_meta = dataset.train_meta
        self.val_meta = dataset.val_meta
        self.test_meta = dataset.test_meta
        self.smap_category = dataset.smap_category
        self.item2meta = dataset.item2meta

        self.num_meta = args.num_meta = len(self.smap_category)
        self.num_items = args.num_items = len(self.smap)
        self.num_users = args.num_users = len(self.train)

        self.max_len = args.trm_max_len
        self.PAD_TOKEN = args.pad_token = self.num_items
        self.META_PAD_TOKEN = args.meta_pad_token = self.num_meta

        args.meta2item = [self.item2meta[i] if i in self.item2meta else 0 for i in range(self.num_items+1)]

        self.test_negative_samples = dataset.negative_samples
        if 'pop' in self.args.model_code:
            self.args.item_freq = self.pop_rec(self.train)
            
    def pop_rec(self, data):
        from collections import Counter
        item_freq = []
        for u in data:
            item_freq.extend(data[u])
        item_freq = Counter(item_freq)
        item_freq_vec = np.zeros(self.num_items)
        for i in item_freq:
            item_freq_vec[i] = item_freq[i]
        return item_freq_vec

    @classmethod
    def code(cls):
        return 'sasrec'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_eval_loader(mode='val')
        test_loader = self._get_eval_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):

        param_dict = {'u2seq': self.train, 'u2seq_meta': self.train_meta, 'max_len': self.max_len, 
                      'num_items': self.num_items, 'num_meta': self.num_meta, 'rng': self.rng, 
                      'pad_token': self.PAD_TOKEN, 'meta_pad_token': self.META_PAD_TOKEN}

        dataset = SASRecTrainDataset(**param_dict)

        dataloader = data_utils.DataLoader(dataset, 
                                           batch_size=self.args.train_batch_size, 
                                           drop_last=False, 
                                           shuffle=True, 
                                           pin_memory=True)

        dataloader.pad_token = self.PAD_TOKEN
        dataloader.meta_pad_token = self.META_PAD_TOKEN
        return dataloader

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        answers = self.val if mode == 'val' else self.test
        answers_meta = self.val_meta if mode == 'val' else self.test_meta

        param_dict = {'u2seq': self.train, 'u2seq_meta': self.train_meta, 'u2answer': answers, 'u2answer_meta':answers_meta, 
                      'max_len': self.max_len, 'negative_samples': self.test_negative_samples, 'pad_token': self.PAD_TOKEN, 
                      'meta_pad_token': self.META_PAD_TOKEN, 'mode': mode, 'val': self.val, 'val_meta': self.val_meta, 'item2meta':self.item2meta}

        dataset = SASRecEvalDataset(**param_dict)

        dataloader = data_utils.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           drop_last=False, 
                                           shuffle=True, 
                                           pin_memory=True)
        return dataloader


class SASRecTrainDataset(data_utils.Dataset):
    def __init__(self, 
                 u2seq, 
                 u2seq_meta, 
                 max_len, 
                 num_items, 
                 num_meta, 
                 rng, 
                 pad_token, 
                 meta_pad_token
                 ):

        self.u2seq = u2seq
        self.u2seq_meta = u2seq_meta
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.num_items = num_items
        self.num_meta = num_meta
        self.pad_token = pad_token
        self.meta_pad_token = meta_pad_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        
        seq = self.u2seq[user]
        seq_category = self.get_seq(self.u2seq_meta[user], 'category')
        labels = seq[-self.max_len-1:]
        category_labels = seq_category[-self.max_len-1:]

        if len(labels) > 1:
            tokens = labels[:-1]
            category_tokens = category_labels[:-1]
            labels = labels[1:]
            category_labels = category_labels[1:]
        else:
            tokens = [self.pad_token]
            category_tokens = [self.meta_pad_token]

        length = len(tokens)

        item_negative = []
        while len(item_negative) < length:
            item_negative_tmp = self.rng.randint(0, self.num_items-1)
            while item_negative_tmp in seq:
                item_negative_tmp = self.rng.randint(0, self.num_items-1)
            item_negative.append(item_negative_tmp)

        category_negative = []
        while len(category_negative) < length:
            category_negative_tmp = self.rng.randint(0, self.num_meta-1)
            while category_negative_tmp in seq_category:
                category_negative_tmp = self.rng.randint(0, self.num_meta-1)
            category_negative.append(category_negative_tmp)

        padding_len = self.max_len - length

        tokens = tokens + [self.pad_token] * padding_len
        category_tokens = category_tokens + [self.meta_pad_token] * padding_len

        labels = torch.LongTensor(labels + [-100] * padding_len).unsqueeze(-1)
        category_labels = torch.LongTensor(category_labels + [-100] * padding_len).unsqueeze(-1)

        negs = torch.LongTensor(item_negative + [-100] * padding_len).unsqueeze(-1)
        category_negs = torch.LongTensor(category_negative + [-100] * padding_len).unsqueeze(-1)

        user = torch.LongTensor([user])
        tokens = torch.LongTensor(tokens)
        meta_tokens = torch.LongTensor(category_tokens)
        candidates = torch.cat((labels, negs), dim=-1)
        meta_candidates = torch.cat((category_labels, category_negs), dim=-1)

        return user, tokens, meta_tokens, candidates, meta_candidates

    def get_seq(self, seq, key):

        return [ele[key] for ele in seq]


class SASRecEvalDataset(data_utils.Dataset):
    def __init__(self, 
                 u2seq, 
                 u2seq_meta, 
                 u2answer, 
                 u2answer_meta,
                 max_len, 
                 negative_samples, 
                 pad_token, 
                 meta_pad_token, 
                 mode, 
                 val, 
                 val_meta,
                 item2meta
                 ):

        self.u2seq = u2seq
        self.u2seq_meta = u2seq_meta
        self.negative_samples = negative_samples
        self.users = {i:u for i,u in enumerate(negative_samples)}
        self.u2answer = u2answer
        self.u2answer_meta = u2answer_meta
        self.max_len = max_len
        self.pad_token = pad_token
        self.meta_pad_token = meta_pad_token
        self.mode = mode
        self.val = val
        self.val_meta = val_meta
        self.item2meta = item2meta
        

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index):
        user = self.users[index]
        user_seq = self.u2seq[user] if self.mode == "val" else self.u2seq[user] + self.val[user]
        user_seq_meta = self.u2seq_meta[user] if self.mode == "val" else self.u2seq_meta[user] + self.val_meta[user]
        seq_category = self.get_seq(user_seq_meta, 'category')
        
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs

        answer_meta = self.get_seq(self.u2answer_meta[user], 'category')
        #negs_meta = [self.item2meta[item] for item in negs]
        #negs_meta.pop(answer_meta[0])
        negs_meta = list(range(self.meta_pad_token))
        negs_meta.pop(answer_meta[0])
        random.shuffle(negs_meta)
        candidates_meta = answer_meta + negs_meta[:100]
        ##

        seq = user_seq[-self.max_len:]
        seq_category = seq_category[-self.max_len:]

        length = len(seq)
        padding_len = self.max_len - length
        seq = seq + [self.pad_token] * padding_len
        seq_category = seq_category + [self.meta_pad_token] * padding_len

        user = torch.LongTensor([user])
        seqs = torch.LongTensor(seq)
        meta_seqs = torch.LongTensor(seq_category)
        candidates = torch.LongTensor(candidates)
        candidates_meta = torch.LongTensor(candidates_meta)
        length = torch.LongTensor([length-1])

        answer = torch.LongTensor(answer)
        answer_meta = torch.LongTensor(answer_meta)

        return user, seqs, meta_seqs, candidates, candidates_meta, length, answer, answer_meta

    def get_seq(self, seq, key):

        return [ele[key] for ele in seq]