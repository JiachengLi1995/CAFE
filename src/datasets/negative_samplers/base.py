from abc import *
import json
import os

class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, item_count, sample_size, seed, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.save_folder = save_folder

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self, save=True):
        if not save:
            return self.generate_negative_samples()
        neg_path = self.get_save_path()
        if not os.path.exists(neg_path):
            negative_samples = self.generate_negative_samples()
            with open(neg_path, 'w') as f:
                json.dump(negative_samples, f)
            return negative_samples
        negative_samples = self.read_json(neg_path)
        return negative_samples

    def get_save_path(self):
        return os.path.join(self.save_folder, "negative-%s-sample-size%d-seed%d.json" % (self.code(), self.sample_size, self.seed))

    def read_json(self, path):
        with open(path, 'r') as f:
            raw = json.load(f)
            data = dict((int(key), value) for (key, value) in raw.items())
            del raw
            return data
