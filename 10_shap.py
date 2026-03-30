"""
Module 10: SHAP Explanations (FINAL FIX)
==============================
Generates SHAP summary plot for the Random Forest model and prints the
top 10 most influential features.
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np

# ── Load model & test set ─────────────────────────────────────────────────────
rf     = joblib.load("rf_model.pkl")
X_test = joblib.load("X_test.pkl")

print("=" * 60)
print("SHAP Explanations — Random Forest")
print("=" * 60)

# ── CRITICAL FIX: Convert numpy array to DataFrame ────────────────────────────
try:
    feature_names = joblib.load("feature_names_clean.pkl")
except:
    feature_names = joblib.load("feature_names.pkl")

print(f"Loaded {len(feature_names)} feature names")
X_test_df = pd.DataFrame(X_test, columns=feature_names)
print(f"X_test converted to DataFrame: {X_test_df.shape}")

# ── Compute SHAP values ───────────────────────────────────────────────────────
sample_size = min(500, len(X_test_df))
X_sample = X_test_df.iloc[:sample_size]

print(f"\nComputing SHAP values for {sample_size} test samples...")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# For binary classification shap_values is list [class0, class1]
if isinstance(shap_values, list):
    sv_class1 = shap_values[1]  # shape usually (n_samples, n_features)
else:
    sv_class1 = shap_values

# Ensure sv_class1 is 2D array of shape (n_samples, n_features)
sv_class1 = np.asarray(sv_class1)

# If shape is (n_features, 2) or similar 2D, we must figure out the layout
if sv_class1.ndim == 2 and sv_class1.shape[1] == 2:
    # This suggests (n_features, 2); assume we want per‑feature statistics across classes
    # Take absolute value, then mean over columns (classes) → 1D per feature
    mean_abs_shap_values = np.abs(sv_class1).mean(axis=1)  # (n_features,)
elif sv_class1.ndim == 2 and sv_class1.shape[0] == 2:
    # This suggests (2, n_features); same fix
    mean_abs_shap_values = np.abs(sv_class1).mean(axis=0)  # (n_features,)
else:
    # Normal case: (n_samples, n_features)
    mean_abs_shap_values = np.abs(sv_class1).mean(axis=0)  # (n_features,)

# ── Summary plot ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
shap.summary_plot(sv_class1, X_sample, show=False, max_display=20)
plt.suptitle("SHAP Summary Plot - Churn Prediction (RF)", fontsize=11, fontweight="bold", y=0.98, ha = 'center')
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔  Saved: shap_summary.png")

# ── FIXED Top influencers ────────────────────────────────────────────────────
# CRITICAL FIX: Ensure 1D array for pandas Series
mean_abs_shap = pd.Series(
    mean_abs_shap_values,  # guaranteed 1D numpy array
    index=X_sample.columns
).sort_values(ascending=False)

print("\n  Top 10 Features by Mean |SHAP| Value:")
for rank, (feat, val) in enumerate(mean_abs_shap.head(10).items(), 1):
    print(f"  {rank:>2}. {feat:<40} {val:.4f}")

print("\n[10_shap.py] ✔ SHAP analysis complete.")