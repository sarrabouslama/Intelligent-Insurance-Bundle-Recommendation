# Intelligent Insurance Bundle Recommendation

Our solution to **DataQuest Phase I**: predict which of 10 insurance coverage bundles a prospective customer will buy, from their demographics, policy history and sales context.

A single LightGBM classifier with a few hand-built features. It runs behind a FastAPI service and a Streamlit dashboard, with Docker and a CI pipeline.

| Holdout macro F1 | Accuracy | Model size | `predict()` on test set (15,218 rows) | Estimated competition score |
| :--------------: | :------: | :--------: | :-----------------------------------: | :-------------------------: |
| **0.561**        | 0.672    | 2.7 MB     | ~2.2 s (1 thread, laptop)              | **~0.43**                   |

---

## The problem in one paragraph

This is a 10-class classification task scored as `macro_f1 × size_penalty × latency_penalty` (see [Scoring](#scoring-recap)). The hard part is the **class imbalance**. Across the 60,868 training rows:

| ID | Bundle                 | Rows   | Share  |
| :- | :--------------------- | -----: | -----: |
| 2  | `Basic_Health`         | 36,136 | 59.4 % |
| 4  | `Health_Dental_Vision` | 13,958 | 22.9 % |
| 3  | `Family_Comprehensive` |  4,831 |  7.9 % |
| 7  | `Premium_Health_Life`  |  2,286 |  3.8 % |
| 1  | `Auto_Liability_Basic` |  1,625 |  2.7 % |
| 0  | `Auto_Comprehensive`   |    823 |  1.4 % |
| 6  | `Home_Standard`        |    719 |  1.2 % |
| 5  | `Home_Premium`         |    479 |  0.8 % |
| 8  | `Renter_Basic`         |      6 | <0.1 % |
| 9  | `Renter_Premium`       |      5 | <0.1 % |

Macro F1 weights each class equally. A model that only learns `Basic_Health` gets about 0.06, and the two renter bundles (11 examples combined) count as much as the majority class.

---

## Approach

### 1. Preprocessing ([src/solution.py](src/solution.py))

- **Missing values:** numeric columns are filled with the median and categorical columns with `"Unknown"`. Most missing values are in `Employer_ID` (94 % missing) and `Broker_ID` (14 %). Region, deductible tier and acquisition channel each have a few hundred.
- **Seasonality:** `Policy_Start_Month` is converted to a number and then to `Month_Sin` / `Month_Cos`, so December and January end up next to each other.
- **Engineered features:**
  - `Total_Deps` = adults + children + infants
  - `Financial_Capacity` = income / (dependents + 1)
  - `Risk_Score` = previous claims × (vehicles + 1)
  - `Policy_Intensity` = previous policy months / (amendments + 1)
- **Categoricals:** the six string columns are label-encoded as integers. LightGBM's tree splits handle these fine, and the result is smaller and faster than one-hot encoding.
- **Dropped:** `User_ID`, `Broker_ID` and `Employer_ID` are high-cardinality IDs, and `Employer_ID` is almost always empty, so they mostly invite overfitting. The raw month string is dropped because the sin/cos pair replaces it.

That leaves **30 features**. `Total_Deps` is the most important one, at about 16 % of total split gain, followed by broker agency type (12 %) and income (10 %). Household composition carries most of the signal, which fits the bundles: family, health and renter products.

### 2. Model ([src/train.py](src/train.py))

```python
LGBMClassifier(
    n_estimators=250, learning_rate=0.07, num_leaves=31,
    min_child_samples=30, min_gain_to_split=0.02,
    class_weight="balanced", colsample_bytree=0.8,
    n_jobs=1, random_state=42,
)
```

- **`class_weight="balanced"`** has the biggest effect on macro F1. It trades majority-class precision for minority-class recall (see results below).
- 80/20 stratified split, with early stopping (patience 40) on the validation fold.
- `n_jobs=1` matches the single-core evaluation container, so our timings are realistic.
- Saved with `joblib.dump(..., compress=9)` to keep the size penalty close to 1.0.

---

## Results

Per-class scores on the 20 % stratified holdout (12,174 rows), using the committed `model.joblib`:

| Bundle                     | Precision | Recall | F1    | Support |
| :------------------------- | --------: | -----: | ----: | ------: |
| 0 `Auto_Comprehensive`     | 0.336     | 0.709  | 0.456 | 165     |
| 1 `Auto_Liability_Basic`   | 0.587     | 0.625  | 0.605 | 325     |
| 2 `Basic_Health`           | 0.896     | 0.721  | 0.799 | 7,227   |
| 3 `Family_Comprehensive`   | 0.319     | 0.643  | 0.427 | 966     |
| 4 `Health_Dental_Vision`   | 0.566     | 0.567  | 0.567 | 2,792   |
| 5 `Home_Premium`           | 0.733     | 0.688  | 0.710 | 96      |
| 6 `Home_Standard`          | 0.400     | 0.625  | 0.488 | 144     |
| 7 `Premium_Health_Life`    | 0.488     | 0.648  | 0.556 | 457     |
| 8 `Renter_Basic`           | 0.000     | 0.000  | 0.000 | 1       |
| 9 `Renter_Premium`         | 1.000     | 1.000  | 1.000 | 1       |
| **Macro avg**              | 0.533     | 0.622  | **0.561** |     |

What this shows:

- **Class weighting works as intended.** Recall is above 0.56 for every class with real support, and recall is higher than precision for almost every minority class. The cost is on `Basic_Health`, where recall drops to 0.72 while precision stays at 0.90.
- **Classes 8 and 9 make macro F1 noisy.** Each has one holdout example, so between them they move the macro average by ±0.1 depending on whether those two rows are right. Don't read too much into small differences between experiments.
- **`Family_Comprehensive` and `Auto_Comprehensive` are the weakest real classes.** Their precision is about 0.33, so the model over-predicts them.

### Where the score goes

| Component          | Value  |
| :----------------- | :----- |
| Macro F1           | 0.561  |
| Size penalty       | 0.987 (2.7 MB) |
| Latency penalty    | ~0.78 (~2.2 s) |
| **Final (est.)**   | **~0.43** |

Model size barely matters. **Latency costs about 22 % of the score.** With 250 trees per class over 10 classes, LightGBM evaluates 2,500 trees per row on a single thread. The timing above is from a laptop and will differ in the grading container.

---

## Trade-offs and decisions

| Decision | Why | Cost |
| :-- | :-- | :-- |
| Gradient-boosted trees instead of a neural net or ensemble | Strong on tabular data, small on disk, no feature scaling, fast to iterate | We skip possible gains from stacking or blending |
| One model rather than an ensemble | Every extra model adds latency, and latency is already our biggest penalty | Probably a few points of F1 |
| `class_weight="balanced"` | Directly targets macro F1 | Lower accuracy and lower majority-class recall |
| Drop `Broker_ID` / `Employer_ID` | High cardinality and mostly missing, a risk of memorising IDs | Could lose real broker-level signal (target encoding is untested) |
| Label encoding instead of one-hot | Fewer columns, faster inference, trees handle integer codes | Encoding depends on the batch (see limitations) |
| `compress=9` | Keeps the size penalty around 0.99 | Slightly slower `load_model()`, which is not timed |


---

## Repository layout

```
src/solution.py        preprocess / load_model / predict (the competition interface)
src/train.py           trains LightGBM, saves model.joblib, prints F1 / size / latency
model.joblib           trained model (2.7 MB)
api/main.py            FastAPI service: POST /predict, GET /info, GET /metrics
frontend/app.py        Streamlit dashboard on top of the API
tests/test_solution.py interface, output-range and latency tests
Dockerfile, docker-compose.yaml
.github/workflows/mlops_ci.yml   flake8 + pytest on every push
data/train.csv, data/test.csv
```

## Running it

```bash
pip install -r requirements.txt

# Retrain (overwrites model.joblib, prints holdout F1, size and latency)
python src/train.py

# Tests
pytest tests/

# API + dashboard
uvicorn api.main:app --port 8000
streamlit run frontend/app.py          # http://localhost:8501

# Or both in containers
docker compose up --build              # API on :8000, dashboard on :8501
```

**API endpoints**

| Endpoint        | Description |
| :-------------- | :---------- |
| `POST /predict` | Upload a CSV with the `test.csv` schema. Returns `predictions.csv` (`User_ID, Purchased_Coverage_Bundle`), with timing in the `X-Prediction-Time-Seconds` header. |
| `GET /info`     | Model type, size, number of trees, feature names, classes. |
| `GET /metrics`  | Macro F1, accuracy, penalties, final score and per-class report, computed on `train.csv`. |

### Competition submission

The graded zip contains `solution.py`, `model.joblib` and a `requirements.txt` at the root. `load_model()` reads `model.joblib` from the working directory.

---

## Scoring recap

```
score = macro_f1 × max(0.5, 1 − model_size_mb / 200) × max(0.5, 1 − predict_seconds / 10)
```

Only `predict()` is timed. The evaluation container has 1 CPU core, 1 GB RAM and a 120 s timeout, and comes with the pinned scikit-learn 1.3.2, LightGBM 4.6.0, XGBoost, CatBoost, PyTorch and TensorFlow. The full column dictionary is in the original challenge brief.
