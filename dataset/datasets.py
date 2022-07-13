import json

import torch
from torch.utils.data import Dataset

from constants import PAIRS_ENCODED


class DataSet(Dataset):

    def __init__(self):
        self.pairs = json.load(open(PAIRS_ENCODED))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])

        return question, reply

    def __len__(self):
        return self.dataset_size
