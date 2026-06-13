import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Directories
DATA_DIR = r"C:\Users\Lenovo\.gemini\antigravity\scratch\d2c_customer_churn\data"
OUTPUT_DIR = r"C:\Users\Lenovo\.gemini\antigravity\scratch\d2c_customer_churn\part3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "rfm_modeling_snapshot.csv"))

# Fill missing categoricals before pipeline
df["loyalty_tier"] = df["loyalty_tier"].fillna("Not Enrolled")

# Split train/val/test using split column
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"Train size: {train_df.shape[0]}")
print(f"Validation size: {val_df.shape[0]}")
print(f"Test size: {test_df.shape[0]}")

# Define features
cat_cols = ["city_tier", "age_group", "acquisition_channel", "loyalty_tier", "preferred_category", "marketing_consent"]
num_cols = [
    "recency_days", "frequency_180d", "monetary_180d", "return_rate_180d", 
    "avg_discount_pct_180d", "avg_rating_180d", "category_diversity_180d", 
    "ticket_count_90d", "negative_ticket_rate_90d", "avg_resolution_hours_90d", 
    "days_since_signup", "sessions_30d", "product_views_30d", "cart_adds_30d", 
    "wishlist_adds_30d", "abandoned_carts_30d", "email_opens_30d", "campaign_clicks_30d", 
    "last_visit_days_ago"
]
target_col = "churn_next_60d"

X_train, y_train = train_df[cat_cols + num_cols], train_df[target_col]
X_val, y_val = val_df[cat_cols + num_cols], val_df[target_col]
X_test, y_test = test_df[cat_cols + num_cols], test_df[target_col]

# Preprocessing Pipeline
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 1. TRAIN BASELINE (Logistic Regression)
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)
y_val_pred_lr = lr_pipeline.predict(X_val)
y_val_prob_lr = lr_pipeline.predict_proba(X_val)[:, 1]

# 2. TRAIN STRONG MODEL (Random Forest)
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
])

rf_pipeline.fit(X_train, y_train)
y_val_pred_rf = rf_pipeline.predict(X_val)
y_val_prob_rf = rf_pipeline.predict_proba(X_val)[:, 1]

# 3. TRAIN SECOND STRONG MODEL (XGBoost)
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="logloss"))
])

xgb_pipeline.fit(X_train, y_train)
y_val_pred_xgb = xgb_pipeline.predict(X_val)
y_val_prob_xgb = xgb_pipeline.predict_proba(X_val)[:, 1]

# Evaluate models on Validation set
def get_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob))
    }

metrics_lr = get_metrics(y_val, y_val_pred_lr, y_val_prob_lr)
metrics_rf = get_metrics(y_val, y_val_pred_rf, y_val_prob_rf)
metrics_xgb = get_metrics(y_val, y_val_pred_xgb, y_val_prob_xgb)

print("\n=== VALIDATION METRICS ===")
print("Logistic Regression (Baseline):", metrics_lr)
print("Random Forest:", metrics_rf)
print("XGBoost:", metrics_xgb)

# Select best model: XGBoost vs Random Forest (compare F1 and AUC)
# Usually XGBoost yields slightly better AUC
best_pipeline = rf_pipeline
best_name = "Random Forest"
metrics_val = metrics_rf
if metrics_xgb["auc"] > metrics_rf["auc"]:
    best_pipeline = xgb_pipeline
    best_name = "XGBoost"
    metrics_val = metrics_xgb

print(f"\nSelected Best Model: {best_name}")

# Evaluate on Test Set
y_test_pred = best_pipeline.predict(X_test)
y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
metrics_test = get_metrics(y_test, y_test_pred, y_test_prob)
print("Test Metrics:", metrics_test)

# Save best model to model.pkl
joblib.dump(best_pipeline, os.path.join(OUTPUT_DIR, "model.pkl"))
print(f"Saved model.pkl to {OUTPUT_DIR}")

# Save metrics.json
metrics_output = {
    "baseline_logistic_regression": {
        "validation": metrics_lr
    },
    "best_model_name": best_name,
    "best_model": {
        "validation": metrics_val,
        "test": metrics_test
    }
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_output, f, indent=4)
print("Saved metrics.json")

# 4. CHOOSE THRESHOLD
# Business Optimization:
# Cost of FP = 150 INR (retention campaign spend on active user)
# Cost of FN = 1250 INR (net lost value from missing a churner)
# Optimal threshold minimizes expected cost = FP_cost * P(active) + FN_cost * P(churn)
# Standard probability threshold of 0.5 can be lowered to capture more recall.
# Let's find the threshold that maximizes F1 or minimizes business cost on validation set.
thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
min_cost = float('inf')
best_val_recall = 0.0
best_val_precision = 0.0

# Calculate optimal business threshold
for t in thresholds:
    y_val_pred_t = (y_val_prob_xgb >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred_t).ravel()
    total_cost = fp * 150 + fn * 1250
    if total_cost < min_cost:
        min_cost = total_cost
        best_threshold = t
        best_val_recall = recall_score(y_val, y_val_pred_t)
        best_val_precision = precision_score(y_val, y_val_pred_t)

print(f"\nOptimal Business Threshold: {best_threshold:.2f}")
print(f"  Validation Cost at 0.50: {confusion_matrix(y_val, y_val_pred_xgb).ravel()[1]*150 + confusion_matrix(y_val, y_val_pred_xgb).ravel()[2]*1250} INR")
print(f"  Validation Cost at {best_threshold:.2f}: {min_cost} INR")
print(f"  Validation Recall at {best_threshold:.2f}: {best_val_recall:.4f}")
print(f"  Validation Precision at {best_threshold:.2f}: {best_val_precision:.4f}")

# 5. ERROR ANALYSIS (False Positives and False Negatives)
test_predictions = pd.DataFrame({
    "customer_id": test_df["customer_id"],
    "true_label": y_test,
    "pred_prob": y_test_prob,
    "pred_label_50": y_test_pred,
    "pred_label_optimal": (y_test_prob >= best_threshold).astype(int)
})

# Merge with features to do analysis
error_df = pd.merge(test_predictions, test_df, on="customer_id")

# False Positives (predicted Churn but stayed, true_label=0, pred_label_50=1)
fps = error_df[(error_df["true_label"] == 0) & (error_df["pred_label_50"] == 1)].head(5)
# False Negatives (predicted Stayed but churned, true_label=1, pred_label_50=0)
fns = error_df[(error_df["true_label"] == 1) & (error_df["pred_label_50"] == 0)].head(5)

print("\n=== SAMPLE FALSE POSITIVES (Model predicted Churn, customer stayed) ===")
print(fps[["customer_id", "pred_prob", "recency_days", "frequency_180d", "monetary_180d", "ticket_count_90d", "last_visit_days_ago"]])

print("\n=== SAMPLE FALSE NEGATIVES (Model predicted Stayed, customer churned) ===")
print(fns[["customer_id", "pred_prob", "recency_days", "frequency_180d", "monetary_180d", "ticket_count_90d", "last_visit_days_ago"]])

# 6. FEATURE IMPORTANCE
# Extract feature names from column transformer
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
all_features = num_cols + cat_features

importances = best_pipeline.named_steps["classifier"].feature_importances_
feat_imp = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== TOP 10 FEATURES ===")
print(feat_imp.head(10).to_string(index=False))

# Save feature importances to CSV
feat_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(15), x="importance", y="feature", palette="viridis")
plt.title("Top 15 Feature Importances (XGBoost Churn Model)")
plt.xlabel("Gini Importance / Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved feature_importance.png")
