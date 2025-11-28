# Loan Default Risk Prediction (End-to-End ML System)
This project implements a complete machine learning workflow for predicting the probability of loan default. It includes data preprocessing, model development, training, evaluation, API serving, and web deployment using Docker and HuggingFace Spaces.  
The system uses two models:
* XGBoost Classifier (primary model)
* MLP Neural Network (PyTorch)

Both models are trained on the cleaned LendingClub dataset and support probability-based predictions.

##
### System Architechture
```
                    ┌─────────────────────────┐
                    │   Raw LendingClub Data  │
                    └──────────────┬──────────┘
                                   │
                         Data Cleaning & Feature
                               Engineering
                     (notebooks/data_preprocess.ipynb)
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │   clean_data.csv        │
                     └──────────────┬──────────┘
                                   │
                 ┌─────────────────┴──────────────────┐
                 │                                    │
                 ▼                                    ▼
      ┌────────────────────┐               ┌────────────────────┐
      │  train_mlp.py      │               │   train_xgb.py     │
      │  (PyTorch MLP)     │               │  (XGBoost)         │
      └─────────┬──────────┘               └─────────┬──────────┘
                │                                    │
                ▼                                    ▼
     ┌────────────────────┐               ┌────────────────────┐
     │  mlp_model.pth     │               │  xgb_model.json    │
     └─────────┬──────────┘               └─────────┬──────────┘
                │                                   │
                └──────────────────┬────────────────┘
                                   ▼
                       ┌───────────────────────┐
                       │  FastAPI Backend      │
                       │      api.py           │
                       └─────────┬─────────────┘
                                 │
                                 ▼
                   ┌─────────────────────────────┐
                   │     Gradio Web Interface    │
                   │            app.py           │
                   └─────────────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────────┐
                      │ HuggingFace Spaces App  │
                      └─────────────────────────┘

```
### Dataset Description  
The project uses the LendingClub accepted loan dataset (2007–2018).  
After cleaning and selection, the final dataset includes these features:
| Feature          | Description                                    |
| ---------------- | ---------------------------------------------- |
| `loan_amnt`      | Loan amount requested                          |
| `term`           | Loan term in months                            |
| `int_rate`       | Interest rate                                  |
| `installment`    | Monthly payment                                |
| `annual_inc`     | Annual income                                  |
| `dti`            | Debt-to-income ratio                           |
| `revol_util`     | Revolving credit utilization                   |
| `open_acc`       | Number of open accounts                        |
| `grade`          | Credit grade (encoded)                         |
| `home_ownership` | Home ownership type (encoded)                  |
| `purpose`        | Loan purpose (encoded)                         |
| `loan_status`    | Original status (string)                       |
| `target`         | Binary target: 1 = Charged Off, 0 = Fully Paid |


The final processed file is stored as:
```
data/clean_data.csv
```

### Project Structure
```
fintech-default-risk/
│
├── data/
│   └── clean_data.csv
│
├── models/
│   ├── mlp_model.pth
│   └── xgb_model.json
│
├── notebooks/
│   └── data_preprocess.ipynb
│
├── src/
│   ├── api.py
│   ├── dataset.py
│   ├── model.py
│   ├── predict.py
│   ├── train_mlp.py
│   ├── train_xgb.py
│   └── test_predict.py
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```
### Installation
Create a virtual environment:
```
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```
Install packages:
```
pip install -r requirements.txt
```

### Data Preprocessing
Run the notebook:
```
notebooks/data_preprocess.ipynb
```
It performs:  
* Feature selection
* Cleaning and type conversion
* Label encoding
* Target engineering
* Saves ```clean_data.csv```

### Train the Models
**XGBoost Training**
```
python src/train_xgb.py
```
Creates:
```
models/xgb_model.json
```
**MLP Training**
```
python src/train_mlp.py
```
Creates:
```
python src/train_mlp.py
```
### Run the API Locally
Starts the server:
```
uvicorn src.api:app --reload
```
Interactive docs:
```
http://127.0.0.1:8000/docs
```

### Run the Gradio App
```
python app.py
```
### Deploy on HuggingFace Spaces
1. Create a Docker Space
2. Upload:
    * ```Dockerfile```
    * ```app.py```
    * ```requirements.txt```
    * ```models/```folder
3. Push the repo
4. HuggingFace auto-builds the app
























### Author  
Rab Nawaz (PhD),  
Machine Learning/Data Science Practitioner


