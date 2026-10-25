"""Week 3: Weather Classification Project."""

import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import requests
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CITY = "Spartanburg, South Carolina"

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

LABEL_RULE = (
    "good=1 when high is 7-26 C inclusive, low >= 0 C, "
    "precipitation < 3 mm, and wind < 30 km/h; otherwise 0"
)


# TASK 1: FETCH WEATHER DATA

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 34.95,
    "longitude": -81.93,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": ",".join(FEATURES),
    "timezone": "America/New_York",
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()["daily"]
df = pd.DataFrame(data)

df["date"] = pd.to_datetime(df["time"])
df = df.drop(columns=["time"])

print("\nTASK 1: WEATHER DATA")
print("City:", CITY)
print("Data shape:", df.shape)
print(df.head())

missing = df[FEATURES].isna().sum()
print("\nMissing values:")
print(missing)

df = df.dropna(subset=FEATURES).copy()

# Missing weather observations are excluded rather than invented.


# TASK 2: CREATE THE RUNNING-DAY LABEL

df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

print("\nTASK 2: CLASS BALANCE")
print(df["good_for_running"].value_counts().sort_index())

good_fraction = df["good_for_running"].mean()
print(f"Fraction of good running days: {good_fraction:.3f}")

# The fraction is calculated from actual downloaded weather data.
# Precision, recall, and F1 help evaluate both classes,
# especially if the class sizes differ.


# TASK 3: TRAIN KNN CLASSIFIER

X = df[FEATURES]
y = df["good_for_running"]

if y.nunique() != 2:
    raise ValueError(
        "Both running-day classes must be present."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTASK 3: KNN CLASSIFIER")

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
k_scores = {}

for k in k_values:
    trial = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    score = cross_val_score(
        trial,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    ).mean()

    k_scores[k] = score
    print(f"k={k}: CV accuracy={score:.4f}")

best_k = max(k_scores, key=k_scores.get)
print("Best k:", best_k)

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(
    y_test, knn_predictions, zero_division=0
)
knn_recall = recall_score(
    y_test, knn_predictions, zero_division=0
)
knn_f1 = f1_score(
    y_test, knn_predictions, zero_division=0
)

print("KNN accuracy:", round(knn_accuracy, 4))
print(classification_report(
    y_test,
    knn_predictions,
    zero_division=0,
))

# StandardScaler is inside the pipeline to prevent
# data leakage during cross-validation.


# TASK 4: TRAIN LOGISTIC REGRESSION

print("\nTASK 4: LOGISTIC REGRESSION")

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_predictions)
log_precision = precision_score(
    y_test, log_predictions, zero_division=0
)
log_recall = recall_score(
    y_test, log_predictions, zero_division=0
)
log_f1 = f1_score(
    y_test, log_predictions, zero_division=0
)

print("Accuracy:", round(log_accuracy, 4))
print("Precision:", round(log_precision, 4))
print("Recall:", round(log_recall, 4))
print("F1:", round(log_f1, 4))

print(classification_report(
    y_test,
    log_predictions,
    zero_division=0,
))

print("\nLogistic regression coefficients:")

for name, coefficient in zip(
    FEATURES,
    log_model.named_steps["classifier"].coef_[0],
):
    print(f"{name}: {coefficient:+.3f}")

# Negative coefficients push toward the skip class.
# Positive coefficients push toward the good class,
# with other standardized features held constant.


# TASK 5: COMPARE MODELS

print("\nTASK 5: MODEL COMPARISON")

print("KNN:")
print("Accuracy:", round(knn_accuracy, 4))
print("Precision:", round(knn_precision, 4))
print("Recall:", round(knn_recall, 4))
print("F1:", round(knn_f1, 4))

print("\nLogistic Regression:")
print("Accuracy:", round(log_accuracy, 4))
print("Precision:", round(log_precision, 4))
print("Recall:", round(log_recall, 4))
print("F1:", round(log_f1, 4))

knn_cm = confusion_matrix(
    y_test, knn_predictions, labels=[0, 1]
)
log_cm = confusion_matrix(
    y_test, log_predictions, labels=[0, 1]
)

print("\nKNN false positives:", int(knn_cm[0, 1]))
print("KNN false negatives:", int(knn_cm[1, 0]))
print("LogReg false positives:", int(log_cm[0, 1]))
print("LogReg false negatives:", int(log_cm[1, 0]))

# False positives recommend running on unsuitable days.
# False negatives miss suitable running opportunities.
# KNN stores training examples and uses distances.
# Logistic regression is compact, interpretable,
# and provides probabilities efficiently.

# Prefer the model with higher test F1.
# Use logistic regression when F1 is tied.

if knn_f1 > log_f1:
    chosen_model = knn_model
    chosen_name = f"KNN (k={best_k})"
    chosen_accuracy = knn_accuracy
    chosen_f1 = knn_f1
else:
    chosen_model = log_model
    chosen_name = "Logistic Regression"
    chosen_accuracy = log_accuracy
    chosen_f1 = log_f1

print("\nChosen model:", chosen_name)


# TASK 6: SAVE MODEL AND METADATA

print("\nTASK 6: SAVE MODEL")

model_path = MODEL_DIR / "weather_classifier.pkl"

metadata_path = (
    MODEL_DIR / "weather_classifier_metadata.json"
)

joblib.dump(chosen_model, model_path)

metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "features": FEATURES,
    "label_rule": LABEL_RULE,
    "city": CITY,
    "model": chosen_name,
    "test_accuracy": float(chosen_accuracy),
    "test_f1": float(chosen_f1),
    "training_days": len(X_train),
    "test_days": len(X_test),
    "best_k": best_k,
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("Saved model:", model_path)
print("Saved metadata:", metadata_path)
print("\nTRAINING COMPLETE")