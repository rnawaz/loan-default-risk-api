import pandas as pd
import torch
from torch.utils.data import Dataset

class LoanDataset(Dataset):
    def __init__(self, features_path, targets_path):

        #load data
        self.X = pd.read_csv(features_path).values.astype('float32')

        y = pd.read_csv(targets_path).values
        self.y = y.reshape(-1).astype('float32')

        #convert to float32 tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]