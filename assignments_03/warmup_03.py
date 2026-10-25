# Week 3: Warmup Exercises

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "weather_classification.csv")

numeric = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# PREPROCESSING - QUESTION 1
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)

print("Scaled training means:", scaled_train.mean(axis=0))

# Fit the scaler on training data only to prevent data leakage.


# PREPROCESSING - QUESTION 2
encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print("Season encoding shape:", season_encoded.shape)
print("Season feature names:", encoder.get_feature_names_out(["season"]))

# One-hot encoding creates one column for each distinct season.


# PREPROCESSING - QUESTION 3
X_with_season = df[numeric + ["season"]]

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_with_season, y, test_size=0.2,
    random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("numbers", StandardScaler(), numeric),
    ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])

season_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000)),
])

season_pipeline.fit(Xs_train, ys_train)

print("Season pipeline accuracy:",
      season_pipeline.score(Xs_test, ys_test))

# The pipeline automatically preprocesses inputs before classification.


# KNN - QUESTION 1
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)

print("KNN accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions, zero_division=0))


# KNN - QUESTION 2
k_scores = {}

for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    trial = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    score = cross_val_score(
        trial, X_train, y_train, cv=5, scoring="accuracy"
    ).mean()

    k_scores[k] = score
    print(f"k={k}: CV accuracy={score:.4f}")

best_k = max(k_scores, key=k_scores.get)

print("Best k:", best_k)

# The chosen k has the highest mean five-fold CV accuracy.


# KNN - QUESTION 3
raw_knn = KNeighborsClassifier(n_neighbors=5)
raw_knn.fit(X_train, y_train)

raw_accuracy = raw_knn.score(X_test, y_test)
scaled_accuracy = knn_pipeline.score(X_test, y_test)

print("Unscaled KNN accuracy:", raw_accuracy)
print("Scaled KNN accuracy:", scaled_accuracy)

# Scaling prevents features with larger numerical ranges
# from dominating distance calculations.


# LOGISTIC REGRESSION - QUESTION 1
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)
log_predictions = pipeline.predict(X_test)

print("LogReg accuracy:", accuracy_score(y_test, log_predictions))
print("LogReg precision:",
      precision_score(y_test, log_predictions, zero_division=0))
print("LogReg recall:",
      recall_score(y_test, log_predictions, zero_division=0))
print("LogReg F1:",
      f1_score(y_test, log_predictions, zero_division=0))


# LOGISTIC REGRESSION - QUESTION 2
coefficients = pipeline.named_steps["classifier"].coef_[0]

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:+.3f}")

skip_features = [
    name for name, coefficient in zip(numeric, coefficients)
    if coefficient < 0
]

print("Features pushing toward skip:", skip_features)

# Negative coefficients push predictions toward skipping a run.


# LOGISTIC REGRESSION - QUESTION 3
classes = list(pipeline.named_steps["classifier"].classes_)
good_idx = classes.index(1)

good_probs = pipeline.predict_proba(X_test)[:, good_idx]

print("First five probabilities:",
      np.round(good_probs[:5], 2))

# A probability near 0.5 means the model is less confident.


# EVALUATION - QUESTION 1
cm = confusion_matrix(y_test, log_predictions, labels=[0, 1])

false_positives = int(cm[0, 1])
false_negatives = int(cm[1, 0])

print("False positives:", false_positives)
print("False negatives:", false_negatives)

ConfusionMatrixDisplay(
    cm, display_labels=["skip", "good"]
).plot()

plt.tight_layout()
plt.savefig(ROOT / "outputs" / "logreg_confusion_matrix.png")
plt.close()

# A false positive recommends running on a day classified
# as unsuitable. A false negative misses a suitable day.


# JOBLIB - QUESTION 1
model_path = ROOT / "models" / "warmup_model.pkl"

joblib.dump(pipeline, model_path)

loaded = joblib.load(model_path)

assert (
    loaded.predict(X_test) == pipeline.predict(X_test)
).all()

print("Predictions match.")

# Saving only the classifier would lose the scaler.
# Saving the complete pipeline preserves preprocessing.ls -lh assignments_03/weather_classification.csv