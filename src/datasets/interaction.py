from .negative_samplers import negative_sampler_factory

import os
import json
import random
import pickle

class ItemDataset(object):
    def __init__(self, args):
        self.args = args
        self.path = args.data_path

        self.train = self.read_json(os.path.join(self.path, "train.json"), True)
        self.val = self.read_json(os.path.join(self.path, "val.json"), True)
        self.test = self.read_json(os.path.join(self.path, "test.json"), True)
        self.data = self.merge(self.train, self.val, self.test)

        self.smap = self.read_json(os.path.join(self.path, "smap.json"))

        self.train_meta = self.read_json(os.path.join(self.path, "train_meta.json"), True)
        self.val_meta = self.read_json(os.path.join(self.path, "val_meta.json"), True)
        self.test_meta = self.read_json(os.path.join(self.path, "test_meta.json"), True)
        self.data_meta = self.merge(self.train_meta, self.val_meta, self.test_meta)

        self.item2meta = self.construct_item2meta(self.data, self.data_meta)

        meta_map = self.read_json(os.path.join(self.path, "smap_meta.json"))
        self.smap_category = meta_map['category']
        #self.smap_status = meta_map['status']

        if self.args.eval_all:
            self.args.metric_ks = [5, 10, 20]
        
        if args.use_pretrained_vectors:
            self.meta = self.read_pickle(os.path.join(self.path, "meta.pickle"))

        negative_sampler = negative_sampler_factory(args.test_negative_sampler_code, self.train, self.val, self.test,
                                                    len(self.smap), args.test_negative_sample_size,
                                                    args.test_negative_sampling_seed, self.path)
        self.negative_samples = negative_sampler.get_negative_samples()

    def merge(self, a, b, c):
        data = {}
        for i in c:
            data[i] = a[i] + b[i] + c[i]
        return data

    def read_json(self, path, as_int=False):
        with open(path, 'r') as f:
            raw = json.load(f)
            if as_int:
                data = dict((int(key), value) for (key, value) in raw.items())
            else:
                data = dict((key, value) for (key, value) in raw.items())
            del raw
            return data

    def construct_item2meta(self, data, data_meta):
        item2meta = dict()
        for user, item_seq in data.items():
            meta_seq = data_meta[user]

            assert len(item_seq) == len(meta_seq)

            for item, meta in zip(item_seq, meta_seq):

                if item in item2meta:
                    assert item2meta[item] == meta['category']
                    continue

                item2meta[item] = meta['category']

        return item2meta

    def read_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def code(cls):
        return "item"