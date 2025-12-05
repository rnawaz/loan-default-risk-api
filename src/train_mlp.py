import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import LoanDataset
from model import MLP


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            total_loss += loss.item()

        return total_loss


def main():
    # load dataset
    dataset = LoanDataset("data/clean_data.csv")

    # train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = dataset.X.shape[1]

    model = MLP(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    epochs = 5
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} "
              f"- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # save the model
    torch.save(model.state_dict(), "models/mlp_model.pth")
    print("MLP model saved to /models/mlp_model.pth")


if __name__ == "__main__":
    main()
