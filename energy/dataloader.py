




import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dde.conditional.r_data_loader import load_dataset

class PrepareData(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PrepareUCIData(Dataset):

    def __init__(self,X):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X).float()
            self.y = torch.zeros([X.shape[0]])
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]