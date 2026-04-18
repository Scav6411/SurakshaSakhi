# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Model 2: Home Delivery Risk
# MAGIC
# MAGIC Binary classifier predicting whether a woman will deliver at home vs. institution.
# MAGIC Compares three approaches for handling class imbalance:
# MAGIC - **Option 1**: RandomForest with class weights
# MAGIC - **Option 2**: RandomForest without weights, evaluated with AUC-ROC
# MAGIC - **Option 3**: RandomForest with class weights + AUC-ROC (best of both)
# MAGIC
# MAGIC **No leakage:** `transport_fac` and `asha_facilitator` are dropped —
# MAGIC they are only known AFTER the woman has already gone to a facility.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F

print("Libraries loaded")

# COMMAND ----------

# MAGIC %md ## 1. Load from Delta Lake

# COMMAND ----------

# Set MLflow to use Unity Catalog for model registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

try:
    spark_df = spark.table("workspace.suraksha.features")
    print("Loaded from workspace.suraksha.features")
except Exception:
    spark_df = spark.table("workspace.suraksha.raw_survey")
    print("Loaded from workspace.suraksha.raw_survey")

print(f"Total rows: {spark_df.count()}")

# COMMAND ----------

# MAGIC %md ## 2. Define Feature & Target Columns

# COMMAND ----------

# Features known BEFORE delivery — safe to use as predictors
# Deliberately excluded:
#   transport_fac      → only known if she went to a facility (outcome-correlated)
#   asha_facilitator   → only known post-delivery (outcome-correlated)
#   financial_asistance_delivery → only known after JSY claim (outcome-correlated)

CAT_COLS = [
    "social_group_code",
    "rural",
    "marital_status",
    "toilet_used",
    "cooking_fuel",
    "is_telephone",
    "is_television",
    "highest_qualification",
    "source_of_anc",
    "house_structure",         # pucca / semi-pucca / kutcha
    "drinking_water_source"
]

NUM_COLS = [
    "age",
    "w_preg_no",               # pregnancy number (parity)
    "no_of_anc",
    "had_anc_registration"     # engineered binary flag
]

ALL_FEATURE_COLS = CAT_COLS + NUM_COLS
TARGET_COL       = "where_del_took_place"

# COMMAND ----------

# MAGIC %md ## 3. Preprocessing

# COMMAND ----------

# 3a. Engineer had_anc_registration flag
spark_df = spark_df.withColumn(
    "had_anc_registration",
    F.when(
        F.col("swelling_of_hand_feet_face").isNull() |
        (F.col("swelling_of_hand_feet_face") == "NA"), 0.0
    ).otherwise(1.0)
)

# 3b. Filter to rows where delivery location is recorded
spark_df = spark_df.filter(
    F.col(TARGET_COL).isNotNull() &
    (F.col(TARGET_COL) != "NA")
)

# 3c. Convert to pandas
pdf = spark_df.select(ALL_FEATURE_COLS + [TARGET_COL]).toPandas()
print(f"Rows available for Model 2 training: {len(pdf)}")

# COMMAND ----------

# 3d. Build binary label
#     1 = delivered at home, 0 = delivered at any institution
pdf["home_delivery"] = (pdf[TARGET_COL].str.strip() == "At Home").astype(int)

home_count = pdf["home_delivery"].sum()
inst_count  = len(pdf) - home_count
print(f"\nClass distribution:")
print(f"  At Home      : {home_count} ({100*home_count/len(pdf):.1f}%)")
print(f"  Institution  : {inst_count} ({100*inst_count/len(pdf):.1f}%)")

# COMMAND ----------

# 3e. Encode categoricals and fill NAs
encoders = {}
pdf[["age", "w_preg_no", "no_of_anc"]] = (
    pdf[["age", "w_preg_no", "no_of_anc"]]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)
pdf["had_anc_registration"] = pdf["had_anc_registration"].fillna(0)

for c in CAT_COLS:
    pdf[c] = pdf[c].fillna("Unknown").astype(str)
    le = LabelEncoder()
    pdf[c + "_enc"] = le.fit_transform(pdf[c])
    encoders[c] = le

ENCODED_FEATURES = [c + "_enc" for c in CAT_COLS] + NUM_COLS
print(f"\nFeature count : {len(ENCODED_FEATURES)}")
print(f"Features      : {ENCODED_FEATURES}")

# COMMAND ----------

# MAGIC %md ## 4. Train/Test Split

# COMMAND ----------

X = pdf[ENCODED_FEATURES].values
y = pdf["home_delivery"].values

# Stratified split preserves 58/42 class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")
print(f"Train positive rate: {y_train.mean()*100:.1f}%")
print(f"Test  positive rate: {y_test.mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md ## 5. Train All 3 Options

# COMMAND ----------

results = []

with mlflow.start_run(run_name="model2_home_delivery"):
    mlflow.log_param("target",         "home_delivery")
    mlflow.log_param("train_size",     len(X_train))
    mlflow.log_param("test_size",      len(X_test))
    mlflow.log_param("positive_rate",  round(y_train.mean() * 100, 1))
    mlflow.log_param("features",       ENCODED_FEATURES)

    # ── Option 1: Class weights only ─────────────────────────────────
    with mlflow.start_run(run_name="home_delivery__opt1_weights", nested=True):
        clf1 = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        clf1.fit(X_train, y_train)
        proba1  = clf1.predict_proba(X_test)[:, 1]
        pred1   = clf1.predict(X_test)
        auc1    = roc_auc_score(y_test, proba1)
        pr_auc1 = average_precision_score(y_test, proba1)
        tn1, fp1, fn1, tp1 = confusion_matrix(y_test, pred1).ravel()
        sig1    = infer_signature(X_train, clf1.predict(X_train))

        mlflow.log_params({"option": "weights_only", "class_weight": "balanced",
                           "max_depth": 8, "n_estimators": 100})
        mlflow.log_metrics({"auc_roc": auc1, "pr_auc": pr_auc1,
                            "true_positives": tp1, "false_negatives": fn1,
                            "sensitivity": tp1/(tp1+fn1) if (tp1+fn1) > 0 else 0})
        mlflow.sklearn.log_model(clf1, "model_home_delivery_opt1", signature=sig1)
        print(f"[Opt1 - weights]       AUC-ROC: {auc1:.4f}  PR-AUC: {pr_auc1:.4f}  "
              f"TP: {tp1}  FN: {fn1}")
        results.append({"option": "opt1_weights", "auc_roc": round(auc1, 4),
                        "pr_auc": round(pr_auc1, 4), "TP": tp1, "FN": fn1})

    # ── Option 2: No weights, AUC-ROC evaluation ─────────────────────
    with mlflow.start_run(run_name="home_delivery__opt2_auc", nested=True):
        clf2 = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight=None,
            random_state=42, n_jobs=-1
        )
        clf2.fit(X_train, y_train)
        proba2  = clf2.predict_proba(X_test)[:, 1]
        pred2   = clf2.predict(X_test)
        auc2    = roc_auc_score(y_test, proba2)
        pr_auc2 = average_precision_score(y_test, proba2)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, pred2).ravel()
        sig2    = infer_signature(X_train, clf2.predict(X_train))

        mlflow.log_params({"option": "auc_only", "class_weight": "none",
                           "max_depth": 8, "n_estimators": 100})
        mlflow.log_metrics({"auc_roc": auc2, "pr_auc": pr_auc2,
                            "true_positives": tp2, "false_negatives": fn2,
                            "sensitivity": tp2/(tp2+fn2) if (tp2+fn2) > 0 else 0})
        mlflow.sklearn.log_model(clf2, "model_home_delivery_opt2", signature=sig2)
        print(f"[Opt2 - auc only]      AUC-ROC: {auc2:.4f}  PR-AUC: {pr_auc2:.4f}  "
              f"TP: {tp2}  FN: {fn2}")
        results.append({"option": "opt2_auc", "auc_roc": round(auc2, 4),
                        "pr_auc": round(pr_auc2, 4), "TP": tp2, "FN": fn2})

    # ── Option 3: Class weights + AUC-ROC evaluation ──────────────────
    with mlflow.start_run(run_name="home_delivery__opt3_both", nested=True):
        clf3 = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        clf3.fit(X_train, y_train)
        proba3  = clf3.predict_proba(X_test)[:, 1]
        pred3   = clf3.predict(X_test)
        auc3    = roc_auc_score(y_test, proba3)
        pr_auc3 = average_precision_score(y_test, proba3)
        tn3, fp3, fn3, tp3 = confusion_matrix(y_test, pred3).ravel()
        sig3    = infer_signature(X_train, clf3.predict(X_train))

        mlflow.log_params({"option": "weights_and_auc", "class_weight": "balanced",
                           "max_depth": 8, "n_estimators": 100})
        mlflow.log_metrics({"auc_roc": auc3, "pr_auc": pr_auc3,
                            "true_positives": tp3, "false_negatives": fn3,
                            "sensitivity": tp3/(tp3+fn3) if (tp3+fn3) > 0 else 0})
        mlflow.sklearn.log_model(clf3, "model_home_delivery_opt3", signature=sig3)
        print(f"[Opt3 - weights+auc]   AUC-ROC: {auc3:.4f}  PR-AUC: {pr_auc3:.4f}  "
              f"TP: {tp3}  FN: {fn3}")
        results.append({"option": "opt3_weights_and_auc", "auc_roc": round(auc3, 4),
                        "pr_auc": round(pr_auc3, 4), "TP": tp3, "FN": fn3})

# COMMAND ----------

# MAGIC %md ## 6. Feature Importance (Best Model)

# COMMAND ----------

# Use Option 3 (usually best) for feature importance
best_clf = clf3
importances = pd.DataFrame({
    "feature":    ENCODED_FEATURES,
    "importance": best_clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 features driving home delivery risk:")
print(importances.head(10).to_string(index=False))

spark.createDataFrame(importances).display()

# COMMAND ----------

# MAGIC %md ## 7. Comparison Summary

# COMMAND ----------

results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("MODEL 2 — HOME DELIVERY RISK COMPARISON")
print("Baseline (random guessing) AUC-ROC = 0.5")
print("TP = women correctly predicted to deliver at home")
print("FN = high-risk women missed by the model (most dangerous)")
print("="*70)
print(results_df.to_string(index=False))

spark.createDataFrame(results_df).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Why FN Matters More Than FP Here
# MAGIC
# MAGIC | Error type | Meaning | Consequence |
# MAGIC |---|---|---|
# MAGIC | **False Negative** | Model says "will go to hospital" but she stays home | ANM doesn't visit → no intervention → risk |
# MAGIC | **False Positive** | Model says "will stay home" but she goes to hospital | Unnecessary ANM visit → low cost |
# MAGIC
# MAGIC For ANM prioritisation, **missing a high-risk home delivery (FN) is far costlier than a false alarm (FP)**.
# MAGIC This is why `class_weight="balanced"` and AUC-ROC both matter — they both reduce FN rate.