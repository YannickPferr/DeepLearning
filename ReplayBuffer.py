 # -*- coding: utf-8 -*-
import random

class ReplayBuffer:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
            
    def sample_idxs(self, sample_size):
        r = range(len(self))

        idxs = random.sample(r, sample_size)
        return idxs
    
    def sample(self, sample_size):
        r = range(len(self))
        idxs = random.sample(r, sample_size)
        sample = []
        for idx in idxs:
            sample.append(self.data[idx])
            
        return sample
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
