# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Model 1: Delivery Complication Risk
# MAGIC
# MAGIC Trains **6 binary classifiers**, one per delivery complication type.
# MAGIC Compares three approaches for handling class imbalance:
# MAGIC - **Option 1**: RandomForest with class weights
# MAGIC - **Option 2**: RandomForest without weights, evaluated with AUC-ROC
# MAGIC - **Option 3**: RandomForest with class weights + AUC-ROC (best of both)
# MAGIC
# MAGIC Reads from Delta Lake → trains with sklearn → logs to MLflow

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from pyspark.sql import functions as F
from pyspark.sql.types  import DoubleType
from mlflow.models.signature import infer_signature

# mlflow.set_experiment("/Workspace/suraksha/model1_complications")
print("MLflow experiment set")

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

# MAGIC %md ## 2. Define Columns

# COMMAND ----------

ANC_SYMPTOM_COLS = [
    "swelling_of_hand_feet_face",
    "hypertension_high_bp",
    "excessive_bleeding",
    "paleness_giddiness_weakness",
    "visual_disturbance",
    "excessive_vomiting",
    "convulsion_not_from_fever"
]

CAT_COLS = [
    "social_group_code", "rural", "toilet_used", "cooking_fuel",
    "is_telephone", "is_television", "highest_qualification", "source_of_anc"
] + ANC_SYMPTOM_COLS

NUM_COLS = ["age", "w_preg_no", "no_of_anc"]

ALL_FEATURE_COLS = CAT_COLS + NUM_COLS + ["had_anc_registration"]

TARGETS = [
    "premature_labour",
    "prolonged_labour",
    "obstructed_labour",
    "excessive_bleeding_during_birth",
    "convulsion_high_bp",
    "breech_presentation"
]

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

# 3b. Fill ANC symptom NAs with "Not_Reported"
for c in ANC_SYMPTOM_COLS:
    spark_df = spark_df.withColumn(
        c,
        F.when(F.col(c).isNull() | (F.col(c) == "NA"), "Not_Reported")
         .otherwise(F.col(c))
    )

# 3c. Filter rows where all delivery complication columns are recorded
spark_df = spark_df.filter(
    F.col("premature_labour").isNotNull() &
    (F.col("premature_labour") != "NA")
)

# 3d. Convert to pandas (only ~4,418 rows — fits in memory)
pdf = spark_df.select(ALL_FEATURE_COLS + TARGETS).toPandas()
print(f"Rows for Model 1 training: {len(pdf)}")

# COMMAND ----------

# 3e. Encode categorical columns with LabelEncoder
#     Fill remaining NAs before encoding
encoders = {}
pdf[NUM_COLS] = pdf[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
pdf["had_anc_registration"] = pdf["had_anc_registration"].fillna(0)

for c in CAT_COLS:
    pdf[c] = pdf[c].fillna("Unknown").astype(str)
    le = LabelEncoder()
    pdf[c + "_enc"] = le.fit_transform(pdf[c])
    encoders[c] = le

ENCODED_FEATURES = [c + "_enc" for c in CAT_COLS] + NUM_COLS + ["had_anc_registration"]

print(f"Feature count: {len(ENCODED_FEATURES)}")
print("Sample feature names:", ENCODED_FEATURES[:5])

# COMMAND ----------

# MAGIC %md ## 4. Train 6 × 3 Combinations

# COMMAND ----------

results = []

with mlflow.start_run(run_name="model1_all_complications"):

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"Target: {target}")

        # Build binary label for this target
        target_series = pdf[target].fillna("No").astype(str)
        y = (target_series == "Yes").astype(int)

        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_rate  = round(100 * pos_count / len(y), 1)
        print(f"  Positive: {pos_count} ({pos_rate}%)  |  Negative: {neg_count}")

        X = pdf[ENCODED_FEATURES].values

        # Stratified split — preserves class ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        row = {"target": target, "positive_rate_%": pos_rate}

        # ── Option 1: Class weights, no AUC focus in training ────────
        with mlflow.start_run(run_name=f"{target}__opt1_weights", nested=True):
            clf1 = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                class_weight="balanced",   # upweights minority class
                random_state=42, n_jobs=-1
            )
            clf1.fit(X_train, y_train)
            proba1   = clf1.predict_proba(X_test)[:, 1]
            auc1     = roc_auc_score(y_test, proba1)
            pr_auc1  = average_precision_score(y_test, proba1)
            sig1     = infer_signature(X_train, clf1.predict(X_train))

            mlflow.log_params({"target": target, "option": "weights_only",
                               "class_weight": "balanced"})
            mlflow.log_metrics({"auc_roc": auc1, "pr_auc": pr_auc1})
            mlflow.sklearn.log_model(
                clf1, f"model_{target}_opt1",
                signature=sig1,
            )
            print(f"  [Opt1 - weights]       AUC-ROC: {auc1:.4f}  PR-AUC: {pr_auc1:.4f}")
            row["opt1_auc_roc"] = round(auc1, 4)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## 5. Comparison Summary

# COMMAND ----------

results_df = pd.DataFrame(results).sort_values("opt3_auc_roc", ascending=False)

print("\n" + "="*75)
print("MODEL 1 — COMPARISON SUMMARY (AUC-ROC, higher = better)")
print("Baseline (random guessing) = 0.5")
print("="*75)
print(results_df.to_string(index=False))

# Render as interactive Databricks table
spark.createDataFrame(results_df).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Interpretation Guide
# MAGIC
# MAGIC | Option | Training | Evaluation | Best when |
# MAGIC |---|---|---|---|
# MAGIC | **Opt 1** | `class_weight="balanced"` | Default | Model needs help finding rare cases |
# MAGIC | **Opt 2** | No weights | AUC-ROC | Imbalance is mild, metric choice matters more |
# MAGIC | **Opt 3** | `class_weight="balanced"` | AUC-ROC | Usually wins — handles both sides |
# MAGIC
# MAGIC **Note:** All three use the same RF architecture — only the weight strategy differs.
# MAGIC Check MLflow Experiments tab for full run history and model artifacts.