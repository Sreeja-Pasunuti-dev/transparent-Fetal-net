import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from modAL.models import ActiveLearner

# 1. Setup
df = pd.read_csv('data/fetal_health.csv')
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'].values.astype(int) - 1

# Start with a very small pool (e.g., 10 samples)
X_initial, y_initial = X.values[:10], y[:10]
X_pool, y_pool = X.values[10:], y[10:]

learner = ActiveLearner(
    estimator=XGBClassifier(),
    X_training=X_initial, y_training=y_initial
)

# 2. Active Learning Loop with XAI
n_queries = 5 

for i in range(n_queries):
    # Find the most uncertain patient
    query_idx, query_inst = learner.query(X_pool)
    
    # --- XAI STEP ---
    explainer = shap.TreeExplainer(learner.estimator)
    shap_vals = explainer.shap_values(query_inst)
    
    # HANDLE DIMENSIONS: Check if shap_vals is a list (multi-class) or array (binary)
    if isinstance(shap_vals, list):
        # If it's a list, we take the last class (usually Pathological or Suspect)
        explanation_to_use = shap_vals[-1][0]
    else:
        # If it's a single array, we take the values directly
        # Sometimes SHAP returns (1, features, classes), we want the features
        if len(shap_vals.shape) == 3:
            explanation_to_use = shap_vals[0, :, -1]
        else:
            explanation_to_use = shap_vals[0]

    # Find the top feature
    top_feature_idx = np.argmax(np.abs(explanation_to_use))
    top_feature_name = X.columns[top_feature_idx]
    
    print(f"Query {i+1}: AI is uncertain about Patient {query_idx[0]}.")
    print(f"🔎 Main source of confusion: {top_feature_name} (Value: {query_inst[0][top_feature_idx]})")
    
    # Teach the learner
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    
    # Update pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    print("--------------------------------------------------")