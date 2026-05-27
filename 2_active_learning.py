import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/fetal_health.csv")

X = df.drop("fetal_health", axis=1).values
y = df["fetal_health"].values.astype(int) - 1

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# 3. Load Previous Active Learning State
# -----------------------------
if os.path.exists("labeled_X.npy"):

    print("Loading previously labeled samples...")

    X_initial = np.load("labeled_X.npy")
    y_initial = np.load("labeled_y.npy")

    X_pool = np.load("pool_X.npy")
    y_pool = np.load("pool_y.npy")

else:

    print("Starting new active learning session...")

    X_initial, X_pool, y_initial, y_pool = train_test_split(
        X_train_full,
        y_train_full,
        train_size=50,
        stratify=y_train_full,
        random_state=42
    )

# -----------------------------
# 4. Initialize Active Learner
# -----------------------------
learner = ActiveLearner(
    estimator=XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    ),
    X_training=X_initial,
    y_training=y_initial
)

# -----------------------------
# 5. Initial Accuracy
# -----------------------------
initial_acc = accuracy_score(y_test, learner.predict(X_test))
print(f"Initial Accuracy: {initial_acc:.4f}")

# -----------------------------
# 6. Active Learning Loop
# -----------------------------
n_queries = 50
received_labels = []
for i in range(n_queries):

    query_idx, query_instance = learner.query(X_pool)
    received_labels.append(y_pool[query_idx][0])  

    # teach model
    learner.teach(
        X=X_pool[query_idx],
        y=y_pool[query_idx]
    )

    # remove queried sample from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    acc = accuracy_score(y_test, learner.predict(X_test))

    print(f"Iteration {i+1}/{n_queries} - Accuracy: {acc:.4f}")
    
for i, label in enumerate(received_labels):
    print(f"Iteration {i+1}: Received label = {label}")
# -----------------------------
# 7. Final Evaluation
# -----------------------------
y_pred = learner.predict(X_test)

final_acc = accuracy_score(y_test, y_pred)

print("\nFinal Model Accuracy:", final_acc)
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Suspect', 'Pathological'], 
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
os.makedirs('tran_imgs', exist_ok=True)
plt.savefig('tran_imgs/confusion_matrix.png')
print("Confusion Matrix saved to tran_imgs/confusion_matrix.png")
plt.close()

# -----------------------------
# 8. Save Model
# -----------------------------
joblib.dump(learner.estimator, "fetal_health_model.pkl")

# -----------------------------
# 9. Save Active Learning State
# -----------------------------
np.save("labeled_X.npy", learner.X_training)
np.save("labeled_y.npy", learner.y_training)

np.save("pool_X.npy", X_pool)
np.save("pool_y.npy", y_pool)

print("\nModel and active learning state saved successfully!")