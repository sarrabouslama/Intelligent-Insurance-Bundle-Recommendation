import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import time
from solution import preprocess, load_model, predict as sol_predict

# ─── Load Data ───
raw_train = pd.read_csv('data/train.csv')
train_processed = preprocess(raw_train)

X = train_processed.drop(columns=['User_ID', 'Purchased_Coverage_Bundle', 'Policy_Start_Month', 'Month_Num', 'Broker_ID', 'Employer_ID'])
y = train_processed['Purchased_Coverage_Bundle']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ─── Smart-Optimized Model ───
model = LGBMClassifier(
    n_estimators=250,        # Sweet spot for accuracy vs size
    learning_rate=0.07,
    num_leaves=31,           
    min_child_samples=30,    
    min_gain_to_split=0.02,  # IMPORTANT: Only adds leaves that actually help
    class_weight='balanced', # Boosts Macro F1 for rare classes
    colsample_bytree=0.8,    # Randomly selects features to prevent overfitting
    random_state=42,
    n_jobs=1,
    verbose=-1
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(40)])

# ─── High Compression Save ───
model_path = 'model.joblib'
joblib.dump(model, model_path, compress=9)

# ─── Verification ───
size_mb = os.path.getsize(model_path) / (1024 * 1024)
val_preds = model.predict(X_val)
macro_f1 = f1_score(y_val, val_preds, average='macro')

# Benchmark
bench_proc = preprocess(raw_train.head(10000))
t0 = time.perf_counter()
sol_predict(bench_proc, model)
latency_s = time.perf_counter() - t0

print(f"\n[RESULTS] Macro F1: {macro_f1:.4f} | Size: {size_mb:.2f}MB | Latency: {latency_s:.3f}s")