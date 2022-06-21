from .base import AbstractNegativeSampler
from tqdm import tqdm
from collections import Counter
import numpy as np

TEST_MAX = 10000
THRESHOLD = 1000000

class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popularity = self.items_by_popularity()

        keys = list(popularity.keys())
        values = [popularity[k] for k in keys]
        sum_value = np.sum(values)
        p = [value / sum_value for value in values]

        negative_samples = {}
        print('Sampling negative items')

        i=0

        for user in tqdm(self.test):
            # if i>TEST_MAX:
            #     break
            i+=1
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

            samples = set()
            while len(samples) < self.sample_size:
                sampled_ids = np.random.choice(keys, self.sample_size*10, replace=False, p=p).tolist()
                for x in sampled_ids:
                    if x not in seen and x not in samples:
                        samples.add(x)
                        if len(samples) == self.sample_size:
                            break
            samples = list(samples)
            negative_samples[user] = samples
        
        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in tqdm(self.test):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        return popularity
