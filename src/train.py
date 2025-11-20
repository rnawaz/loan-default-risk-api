import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LoanDataset
from model import MLP

def train():
    
    # paths to the processed data
    train_features = "data/X_train.csv"
    train_targets = "data/y_train.csv"

    # create dataset and dataloader
    dataset = LoanDataset(train_features, train_targets)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    # model
    num_features = dataset.X.shape[1]
    model = MLP(num_features)

    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    epochs = 5
    for epoc in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            preds = model(X).squeeze()

            # compute loss
            loss = criterion(preds,y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoc+1}/{epochs}, Loss: {total_loss:.4f}")

    # save the model
    torch.save(model.state_dict(), "models/best_model.pth")
    print("Model saved to ../models/best_model.pth")

if __name__=="__main__":
    train()
