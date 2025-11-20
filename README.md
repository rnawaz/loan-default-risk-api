# Loan Default Risk Prediction API  
An end-to-end machine learning system for predicting loan default probability.
The project includes a PyTorch model, FastAPI backend, Gradio user interface, Docker container, and deployment on HuggingFace Spaces.
This simulates a realistic ML workflow used in lending and fintech environments.
##
### Project Features  
**1. Model Training (PyTorch)**
* Uses a processed Lending Club-style dataset
* Feed-forward neural network (MLP)
* Saved model weights stored in models/best_model.pth
* Configurable number of features

**2. API (FastAPI)**  
A production-style prediction endpoint located at */predict*.  
Receives a JSON payload and returns a default probability.

**Example request:**

```
{
  "loan_amnt": 10000,
  "term": 36,
  "int_rate": 12.5,
  "installment": 310,
  "annual_inc": 65000,
  "dti": 14.0,
  "revol_util": 45.0,
  "open_acc": 6,
  "grade": 2,
  "home_ownership": 1,
  "purpose": 4
}
```

**Example response:**

```
{
  "default_probability": 0.387
}
```

**3. Gradio Web Interface**  
A simple interactive UI that allows users to enter loan features and view predicted default risk.

**4. Docker Deployment**  
The entire project is containerized using a custom Dockerfile and deployed to HuggingFace Spaces.

##
### Project Structure
```
loan-default-risk-api/
│
├── app.py                  # Gradio UI application
├── requirements.txt
├── Dockerfile
│
├── src/
│   ├── api.py              # FastAPI prediction service
│   ├── model.py            # PyTorch model definition
│   ├── predict.py          # Preprocessing and inference helper
│   ├── train.py            # Training script (optional)
│   ├── dataset.py          # Dataset utilities (optional)
│   └── __init__.py
│
├── models/
│   └── best_model.pth      # Trained model weights
│
└── README.md

```
##
### Running the Project Locally  
1. Create and activate a virtual environment
```
python -m venv venv
```

Windows:
```
venv\Scripts\activate
```

macOS/Linux:
```
source venv/bin/activate
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Run the FastAPI backend**
```
uvicorn src.api:app --reload
```
The interactive API documentation will be available at:
```
http://127.0.0.1:8000/docs
```

**4. Run the Gradio UI**
```
python app.py
```
##
### Deployment (HuggingFace Spaces)
The project uses a custom Dockerfile:
```
FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

EXPOSE 7860

CMD ["python", "app.py"]
```
HuggingFace automatically builds and runs the container.
##
### Future Improvements
* Improved data preprocessing
* Feature scaling and encoding
* Enhanced model architecture
* Evaluation metrics and validation
* Risk categorization (low, medium, high)
* Deployment to additional/cloud platforms (AWS)

### Author  
Rab Nawaz (PhD),  
Machine Learning/Data Science Practitioner


