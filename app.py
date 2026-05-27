import numpy as np
import joblib
import shap
from flask import Flask, render_template, request, jsonify
from modAL.models import ActiveLearner
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json
import base64
from datetime import datetime

app = Flask(__name__)

# Load entire dataset to maintain a test set for live accuracy calculation
df = pd.read_csv("data/fetal_health.csv")
X = df.drop("fetal_health", axis=1).values
y = df["fetal_health"].values.astype(int) - 1

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load saved active learning state
X_initial = np.load("labeled_X.npy")
y_initial = np.load("labeled_y.npy")

X_pool = np.load("pool_X.npy")
y_pool = np.load("pool_y.npy")

# Recreate learner
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

# Initial size is hardcoded as 50 based on 2_active_learning.py initial split
INITIAL_TRAINING_SIZE = 50

# Track progress for the dashboard
def load_al_state():
    state_file = "data/al_state.json"
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {
        "iteration": len(X_initial) - INITIAL_TRAINING_SIZE,
        "queried_samples": len(X_initial) - INITIAL_TRAINING_SIZE
    }

def save_al_state(state):
    state_file = "data/al_state.json"
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f)

al_state = load_al_state()
iteration = al_state.get("iteration", len(X_initial) - INITIAL_TRAINING_SIZE)
queried_samples = al_state.get("queried_samples", len(X_initial) - INITIAL_TRAINING_SIZE)

current_train_size = len(learner.X_training)
if current_train_size - INITIAL_TRAINING_SIZE > queried_samples:
    queried_samples = current_train_size - INITIAL_TRAINING_SIZE
    iteration = queried_samples
    save_al_state({"iteration": iteration, "queried_samples": queried_samples})

progress_history = [accuracy_score(y_test, learner.predict(X_test)) * 100]

feature_names = [
    'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_long_term_variability',
    'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
    'histogram_variance', 'histogram_tendency'
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns the current state of the AL model and the next queried sample."""
    global X_pool, y_pool, learner, iteration, queried_samples, progress_history
    
    if len(X_pool) == 0:
        return jsonify({"error": "No more samples in the pool"}), 400

    # Current model accuracy
    acc = accuracy_score(y_test, learner.predict(X_test)) * 100
    
    # Query sample
    query_idx, query_instance = learner.query(X_pool)
    # modAL returns the array of indices as the first item
    query_idx = query_idx[0]
    X_current = X_pool[query_idx]
    
    # Prediction probabilities
    proba = learner.predict_proba(X_current.reshape(1, -1))[0]
    
    # SHAP explanation (native XGBoost pred_contribs)
    pred_class = int(learner.predict(X_current.reshape(1, -1))[0])
    contribs = learner.estimator.get_booster().predict(xgb.DMatrix(X_current.reshape(1, -1)), pred_contribs=True)
    importance = contribs[0, pred_class, :-1]
    
    return jsonify({
        "iteration": iteration,
        "accuracy": acc,
        "queriedSamples": queried_samples,
        "initialTrainingSize": INITIAL_TRAINING_SIZE,
        "trainSize": len(learner.X_training),
        "remainingPool": len(X_pool),
        "progressHistory": progress_history,
        "sample": {
            "features": X_current.tolist(),
            "featureNames": feature_names
        },
        "probabilities": proba.tolist(),
        "shap": {
            "values": importance.tolist()
        }
    })

@app.route("/api/submit", methods=["POST"])
def submit_label():
    """Receives human label, trains the model, and removes from pool."""
    global X_pool, y_pool, learner, iteration, queried_samples, progress_history
    
    data = request.json
    label_str = data.get("label")
    
    # Map back to integer label based on UI
    label_map = {"Normal": 0, "Suspect": 1, "Pathological": 2}
    if label_str not in label_map:
        return jsonify({"error": "Invalid label"}), 400
        
    label = label_map[label_str]
    
    # Re-query to find the index we're training on
    query_idx, query_instance = learner.query(X_pool)
    query_idx = query_idx[0]
    X_current = X_pool[query_idx]
    
    # Teach the learner
    learner.teach(X_current.reshape(1, -1), np.array([label]))
    
    # Remove from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    # Update metrics
    iteration += 1
    queried_samples += 1
    save_al_state({"iteration": iteration, "queried_samples": queried_samples})
    
    new_acc = accuracy_score(y_test, learner.predict(X_test)) * 100
    progress_history.append(new_acc)
    
    # Save the new state natively
    np.save("labeled_X.npy", learner.X_training)
    np.save("labeled_y.npy", learner.y_training)
    np.save("pool_X.npy", X_pool)
    np.save("pool_y.npy", y_pool)
    joblib.dump(learner.estimator, "fetal_health_model.pkl")
    
    return jsonify({"success": True})

@app.route("/ctg_analysis", methods=["GET"])
def ctg_analysis():
    return render_template("ctg_analysis.html")

@app.route("/api/analyze_ctg", methods=["POST"])
def analyze_ctg():
    data = request.json
    feature_dict = data.get("features", {})
    
    # Map to array based on feature_names order
    x_input = []
    for f_name in feature_names:
        val = feature_dict.get(f_name, 0.0)
        if val == "":
            val = 0.0
        x_input.append(float(val))
        
    x_input_arr = np.array([x_input])
    
    # Model Loader Integration
    model_to_use = learner.estimator
    if os.path.exists("fetal_health_model.pkl"):
        try:
            model_to_use = joblib.load("fetal_health_model.pkl")
        except Exception as e:
            print("Fallback to active learner memory model.")
            
    # Predict
    pred_idx = int(model_to_use.predict(x_input_arr)[0])
    proba = model_to_use.predict_proba(x_input_arr)[0].tolist()
    
    # SHAP explanation (native XGBoost pred_contribs)
    contribs = model_to_use.get_booster().predict(xgb.DMatrix(x_input_arr), pred_contribs=True)
    shap_vals = contribs[0, pred_idx, :-1].tolist()
    
    # Formatting to match requested spec and align Risk Level with Status intrinsically
    confidence = proba[pred_idx] * 100
    
    if pred_idx == 2:
        risk_level = "High Risk"
        final_label = "Pathological"
        final_idx = 2
    elif pred_idx == 1:
        risk_level = "Medium Risk"
        final_label = "Suspect"
        final_idx = 1
    else: # Normal Prediction
        if confidence > 85:
            risk_level = "Low Risk"
            final_label = "Normal"
            final_idx = 0
        elif confidence >= 60:
            risk_level = "Medium Risk"
            final_label = "Suspect"
            final_idx = 1
        else:
            risk_level = "High Risk"
            final_label = "Pathological"
            final_idx = 2
            
    if final_label == "Normal":
        clinical_summary = "The CTG data indicates a normal fetal condition. Routine monitoring is recommended."
        suggested_action = "Continue Routine Monitoring"
    elif final_label == "Suspect":
        clinical_summary = "The CTG data indicates a suspect fetal condition. Close medical monitoring is advised."
        suggested_action = "Schedule Medical Review"
    else:
        clinical_summary = "The CTG data indicates a pathological fetal condition. Immediate medical attention is required."
        suggested_action = "Immediate Medical Intervention Required"
        
    feature_shaps = [{"feature": feature_names[i].replace('_', ' ').title(), "value": shap_vals[i]} for i in range(len(feature_names))]
    feature_shaps.sort(key=lambda x: abs(x["value"]), reverse=True)
    top_features = feature_shaps[:5]
    
    return jsonify({
        "prediction": final_label,
        "confidence": confidence,
        "risk_level": risk_level,
        "top_features": top_features,
        "clinical_summary": clinical_summary,
        "suggested_action": suggested_action,
        "raw_prediction_idx": final_idx,
        "probabilities": proba
    })

@app.route("/api/save_report", methods=["POST"])
def save_report():
    data = request.json
    data['timestamp'] = datetime.now().isoformat()
    
    reports_file = "data/reports.json"
    os.makedirs(os.path.dirname(reports_file), exist_ok=True)
    
    reports = []
    if os.path.exists(reports_file):
        with open(reports_file, "r") as f:
            try:
                content = f.read()
                if content:
                    reports = json.loads(content)
            except Exception:
                pass
                
    # Insert new at top
    reports.insert(0, data)
    
    with open(reports_file, "w") as f:
        json.dump(reports, f)
        
    return jsonify({"success": True})

@app.route("/api/get_reports", methods=["GET"])
def get_reports():
    reports_file = "data/reports.json"
    if os.path.exists(reports_file):
        with open(reports_file, "r") as f:
            try:
                return jsonify(json.load(f))
            except Exception:
                return jsonify([])
    return jsonify([])

@app.route("/api/delete_report", methods=["POST"])
def delete_report():
    timestamp = request.json.get("timestamp")
    reports_file = "data/reports.json"
    if os.path.exists(reports_file):
        try:
            with open(reports_file, "r") as f:
                reports = json.load(f)
            
            new_reports = [r for r in reports if r.get("timestamp") != timestamp]
            
            with open(reports_file, "w") as f:
                json.dump(new_reports, f)
                
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False}), 404

@app.route("/reports", methods=["GET"])
def reports_view():
    return render_template("reports.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)