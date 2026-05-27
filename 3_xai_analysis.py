import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 1. Load data
df = pd.read_csv('data/fetal_health.csv')
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'].values.astype(int) - 1

# 2. Train the model
model = XGBClassifier()
model.fit(X, y)

# 3. Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# --- PART 1: Global Summary (The Bar Chart) ---
plt.figure(figsize=(10, 6))
# This shows the most important features for all classes
shap.summary_plot(shap_values, X, plot_type="bar", class_names=["Normal", "Suspect", "Pathological"], show=False)
plt.title("Global Medical Feature Importance")
plt.show()
# --- PART 2: Local Explanation (The Browser File) ---
patient_index = 0 

# Use .iloc[patient_index].values to ensure it's a clean array of numbers
feature_values = X.iloc[patient_index].values
feature_names = X.columns

# We use shap_values[2] for the "Pathological" class
plot = shap.force_plot(
    explainer.expected_value[2], 
    shap_values[patient_index, :, 2],
    features=feature_values,
    feature_names=feature_names
)

# Save the interactive version
shap.save_html("patient_explanation.html", plot)

print(f"✅ Local explanation for Patient {patient_index} saved to 'patient_explanation.html'.")