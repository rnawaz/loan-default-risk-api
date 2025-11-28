import gradio as gr
import torch
import pandas as pd
from src.predict import (load_mlp, preprocess_input, predict_mlp, load_xgb, predict_xgb)

FEATURE_ORDER = [
    "loan_amnt", "term", "int_rate", "installment",
    "annual_inc", "dti", "revol_util", "open_acc",
    "grade", "home_ownership", "purpose"
]

# load models
MLP_MODEL_PATH = "models/mlp_model.pth"
XGB_MODEL_PATH = "models/xgb_model.json"

mlp_model = load_mlp(MLP_MODEL_PATH, len(FEATURE_ORDER))
xgb_model = load_xgb(XGB_MODEL_PATH)


# ----------------------------------------------------
# Unified prediction function for Gradio
# ----------------------------------------------------
def prediction_interface(
        loan_amnt, term, int_rate, installment, annual_inc,
        dti, revol_util, open_acc, grade, home_ownership, purpose,
        model_type
):
    # build inpput dict
    input_dict = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "revol_util": revol_util,
        "open_acc": open_acc,
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose
    }

    if model_type=="MLP":
        tensor = preprocess_input(input_dict, FEATURE_ORDER)
        prob = predict_mlp(mlp_model, tensor)

    else:
        prob = predict_xgb(xgb_model, input_dict, FEATURE_ORDER)

    return f"Default Probability: {prob:0.4f}"


# ----------------------------------------------------
# Gradio UI
# ----------------------------------------------------
inputs = [
    gr.Number(label="Loan Amount"),
    gr.Number(label="Term (months)"),
    gr.Number(label="Interest Rate"),
    gr.Number(label="Installment"),
    gr.Number(label="Annual Income"),
    gr.Number(label="DTI"),
    gr.Number(label="Revolving Util (%)"),
    gr.Number(label="Open Acounts"),
    gr.Number(label="Grade (encoded)"),
    gr.Number(label="Home Ownership(encoded"),
    gr.Number(label="Purpose (encoded)"),
    gr.Radio(["MLP", "XGBoost"], label="Model", value="XGBoost")
]

output = gr.Textbox(label="Prediction")

app = gr.Interface(
    fn=prediction_interface,
    inputs=inputs,
    outputs=output,
    title="Loan Default Risk Prediction",
    description="Choose model and enter loan features to predict default probability"
)

if __name__=="__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)