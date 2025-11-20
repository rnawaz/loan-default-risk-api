from fastapi import FastAPI
from pydantic import BaseModel
import torch

from .model import MLP
from .predict import preprocess_input, predict


#--------------- Feature oredr (as iin training) ----------
feature_oreder = ['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti',
                   'revol_util', 'open_acc', 'grade', 'home_ownership', 'purpose']


#--------------- Load the model -------------------------
num_features = len(feature_oreder)
model_path = "models/best_model.pth"

model = MLP(num_features)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

#-------------- FastAPI app -----------------------------
app = FastAPI()
class LoanApplication(BaseModel):
    loan_amnt: float
    term: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    revol_util: float
    open_acc: float
    grade: float
    home_ownership: float
    purpose: float

@app.post("/predict")
def predict_default(data: LoanApplication):
    input_dict = data.model_dump()
    input_tensor = preprocess_input(input_dict, feature_oreder)
    prob = predict(model, input_tensor)
    return {"default_probability": prob}
