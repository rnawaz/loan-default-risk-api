import torch
import pandas as pd
from .model import MLP


def load_model(model_path, num_features):
    model = MLP(num_features)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_input(input_dict, feature_order):
    # convert dictionary to a list in the correct order
    values = []
    for feat in feature_order:
        values.append(float(input_dict[feat]))
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

def predict(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
    return prob
