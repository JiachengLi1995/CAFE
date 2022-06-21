from .base import AbstractNegativeSampler
from tqdm import tqdm
import numpy as np
import random

TEST_MAX = 100000
THRESHOLD = 1000000

class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}

        print('Sampling negative items')

        if self.item_count < THRESHOLD:
            for user in tqdm(self.test):
                if isinstance(self.train[user][0], tuple):
                    seen = set(x[0] for x in self.train[user])
                    seen.update(x[0] for x in self.val[user])
                    seen.update(x[0] for x in self.test[user])
                elif isinstance(self.train[user][0], dict):
                    seen = set(x['category'] for x in self.train[user])
                    seen.update(x['category'] for x in self.val[user])
                    seen.update(x['category'] for x in self.test[user])
                else:
                    seen = set(self.train[user])
                    seen.update(self.val[user])
                    seen.update(self.test[user])

                samples = []
                for _ in range(self.sample_size):
                    item = np.random.choice(self.item_count)
                    cnt = 0
                    while item in seen or item in samples:
                        if cnt > 100: break
                        item = np.random.choice(self.item_count)
                        cnt += 1
                    samples.append(item)

                negative_samples[user] = samples

        else:
            # fast sampling --> assuming that the prob of sampling postive items from a large items pool is 0
            candidates = range(self.item_count)
            test = list(self.test.keys())
            random.shuffle(test)

            for i, user in tqdm(enumerate(test)):
                if i > TEST_MAX: 
                    break
                negative_samples[user] = random.choices(candidates, k=self.sample_size)

        return negative_samples
