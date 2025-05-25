import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("claimflowengine/data/processed_claims.csv")
y = df["denied"]
X = df.drop(columns=["denied"])

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)

# Explain
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Save bar plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance (XGBoost)")
plt.tight_layout()
plt.savefig("logs/shap_feature_importance.png")
print("SHAP plot saved to logs/shap_feature_importance.png")
