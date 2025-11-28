import pandas as pd
import torch
from torch.utils.data import Dataset

class LoanDataset(Dataset):
    def __init__(self, csv_path: str):

        # load data
        df = pd.read_csv(csv_path)

        self.X = df.drop(columns=["target"]).values.astype("float32") # features
        self.y = df["target"].values.astype("float32")                # targets

        #convert to tensors
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]