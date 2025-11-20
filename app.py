import gradio as gr
import torch
import pandas as pd
from src.model import MLP

# load model
model_path = "models/best_model.pth"
model = MLP(num_features=11)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# prediction function
def predict_ui(loan_amnt, term, int_rate, installment, annual_inc, dti,
               revol_util, open_acc, grade, home_ownership, purpose):
    
    # create input dataframe for the model
    data = pd.DataFrame([[
        loan_amnt, term, installment, annual_inc, dti,
        revol_util, open_acc, grade, home_ownership, purpose
    ]])

    # convert to tensor
    x = torch.tensor(data.values).float()

    with torch.no_grad():
        prob = model(x).item()

    return f"{prob:.3f}"

# build UI
ui = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Number(label="Loan Amount"),
        gr.Number(label="Term (months)"),
        gr.Number(label="Interest Rate (%)"),
        gr.Number(label="Installment"),
        gr.Number(label="Annual Income"),
        gr.Number(label="DTI"),
        gr.Number(label="Revolving Util (%)"),
        gr.Number(label="Open Accounts"),
        gr.Number(label="Grade (encoded)"),
        gr.Number(label="Home Ownership (encoded)"),
        gr.Number(label="Purpose (encoded)")
    ],
    outputs=gr.Textbox(label="Dafault Probability"),
    title="Loan Default Prediction",
    description="Enter loan detials to get the predicted default probability"
)

# Launcg for HUggingFace Spaces
ui.launch(server_name="0.0.0.0", server_port=7860)