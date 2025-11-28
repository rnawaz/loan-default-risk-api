import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# load the dataset
df = pd.read_csv("data/clean_data.csv")

# feature columns
feature_cols = [
    "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
    "revol_util", "open_acc", "grade", "home_ownership", "purpose"
]

X = df[feature_cols]
y = df["target"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y
                                                    )
# train xgboost
model = xgb.XGBClassifier(
    n_estimators=300,       # num of trees
    max_depth=6,            # depth of each tree
    learning_rate=0.05,     # step size
    subsample=0.8,          # row sampling
    colsample_bytree=0.8,   # feature sampling
    objective="binary:logistic",   # binary classification
    eval_metric="auc",
    random_state=42,
    n_jobs=1
)

print("Training XGBoost model...")
model.fit(X_train,y_train)

# evaluate
preds = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, preds)
print(f"\nAUC Score: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, preds > 0.5))

# save the model
model.save_model("models/xgb_model.json")
print("\nModel saved to /models/xgb_model.json")