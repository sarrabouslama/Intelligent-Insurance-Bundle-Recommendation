"""train.py — Train and save the model. Optimized for Macro F1 + small size + fast inference."""
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from solution import preprocess

# ─── Load and preprocess ─────────────────────────────────────────────────────
raw_train = pd.read_csv('data/train.csv')
train_processed = preprocess(raw_train)

X = train_processed.drop(columns=['User_ID', 'Purchased_Coverage_Bundle', 'Policy_Start_Month', 'Month_Num', 'Broker_ID', 'Employer_ID'])
y = train_processed['Purchased_Coverage_Bundle']

print(f"Features ({len(X.columns)}): {list(X.columns)}")
print(f"Classes: {sorted(y.unique())}")
print(f"Samples: {len(y)}")

# ─── Stratified split ────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── Model ───────────────────────────────────────────────────────────────────
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.08,
    num_leaves=31,
    min_child_samples=20,
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

# ─── Train ───────────────────────────────────────────────────────────────────
print("\nTraining...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[
        lgb.early_stopping(50, verbose=True),
        lgb.log_evaluation(50),
    ]
)

print(f"\nBest iteration: {model.best_iteration_}")

# ─── Evaluate ────────────────────────────────────────────────────────────────
preds = model.predict(X_val)
macro_f1 = f1_score(y_val, preds, average='macro')

print(f"\nLocal Validation Macro F1: {macro_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_val, preds, zero_division=0))

# ─── Save ────────────────────────────────────────────────────────────────────
joblib.dump(model, 'model.joblib', compress=5)
import os
size_mb = os.path.getsize('model.joblib') / (1024 * 1024)
print(f"\nModel saved: {size_mb:.2f} MB")

# ─── Quick latency check ────────────────────────────────────────────────────
import time
from solution import load_model, predict as sol_predict
test_sample = train_processed.head(10000)
m = load_model()
t0 = time.perf_counter()
sol_predict(test_sample, m)
latency = time.perf_counter() - t0
print(f"Predict latency (10k rows): {latency:.3f}s")