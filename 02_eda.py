"""
Module 02: Exploratory Data Analysis
=====================================
Generates visualisations and prints key insights from the Telco Churn dataset.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

churn_rate = (df["Churn"] == "Yes").mean()
print("=" * 60)
print("Exploratory Data Analysis")
print("=" * 60)
print(f"\nOverall Churn Rate : {churn_rate:.2%}")

# ── Palette ───────────────────────────────────────────────────────────────────
palette = {"Yes": "#e74c3c", "No": "#2ecc71"}
sns.set_theme(style="whitegrid", palette="muted")

# ── Plot 1: Contract type vs Churn (saved as eda_tenure.png) ──────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Contract", hue="Churn", data=df, palette=palette, ax=ax)
ax.set_title("Churn by Contract Type", fontsize=12, fontweight="bold")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Customer Count")
ax.legend(title="Churn")
plt.tight_layout()
plt.savefig("eda_tenure.png", dpi=150)
plt.close()
print("\n  ✔  Saved: eda_tenure.png  (Contract × Churn)")

# ── Plot 2: Monthly Charges vs Churn (saved as eda_charges.png) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=palette, ax=axes[0])
axes[0].set_title("Monthly Charges by Churn", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Monthly Charges ($)")

# Tenure distribution
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, palette=palette,
             bins=30, ax=axes[1], element="step")
axes[1].set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_charges.png", dpi=150)
plt.close()
print("  ✔  Saved: eda_charges.png  (MonthlyCharges & Tenure × Churn)")

# ── Plot 3: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, square=True, cbar_kws={"shrink": 0.8})
ax.set_title("Numeric Feature Correlation Heatmap", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_heatmap.png", dpi=150)
plt.close()
print("  ✔  Saved: eda_heatmap.png  (Correlation Heatmap)")

# ── Key Insights ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Key Insights")
print("=" * 60)

# Insight 1 — Contract type
contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean()
).sort_values(ascending=False)
print(
    f"\n1. Month-to-month customers churn the most "
    f"({contract_churn.iloc[0]:.1%}), while two-year contracts "
    f"show the lowest churn ({contract_churn.iloc[-1]:.1%})."
)

# Insight 2 — Monthly charges
mean_charges = df.groupby("Churn")["MonthlyCharges"].mean()
print(
    f"\n2. Churned customers pay significantly higher monthly charges on average "
    f"(${mean_charges['Yes']:.2f}) vs non-churned (${mean_charges['No']:.2f})."
)

# Insight 3 — Tenure
mean_tenure = df.groupby("Churn")["tenure"].mean()
print(
    f"\n3. Churned customers have much shorter tenures on average "
    f"({mean_tenure['Yes']:.1f} months) vs retained customers "
    f"({mean_tenure['No']:.1f} months), highlighting early-stage vulnerability."
)

print("\n[02_eda.py] ✔ EDA complete — plots saved.")
