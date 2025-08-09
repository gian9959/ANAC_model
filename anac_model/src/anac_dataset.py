import random
import numpy as np
import pymysql as pymysql
from torch.utils.data import Dataset

import pandas


class AnacDataset(Dataset):
    def __init__(self, raw_data):

        self.data = []
        for d in raw_data:
            random.shuffle(d["companies"])

            tender = dict(d)
            self.data.append(tender)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
