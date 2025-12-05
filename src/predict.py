import torch
import pandas as pd
import xgboost as xgb
from .model import MLP


# load MLP
def load_mlp(model_path, num_features):
    model = MLP(num_features)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# load xgboost
def load_xgb(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


# preprocess input
def preprocess_input(input_dict, feature_order):
    # convert dictionary to list in correct order
    values = []

    for feat in feature_order:
        values.append(float(input_dict[feat]))

    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


# -----  Predict with MLP -----
def predict_mlp(model, input_tensor):
    with torch.no_grad():
        prob = model(input_tensor).item()
    return float(prob)


# ----- Predict with XGBoost -----
def predict_xgb(model, input_dict, feature_order):
    row = []
    for feat in feature_order:
        row.append(float(input_dict[feat]))
    df = pd.DataFrame([row], columns=feature_order)
    prob = model.predict_proba(df)[0][1]
    return float(prob)
