# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Model 3: Immunization Dropout Risk
# MAGIC
# MAGIC Binary classifier predicting whether a child will have an incomplete
# MAGIC immunization schedule, using **household and maternal characteristics only**.
# MAGIC
# MAGIC **No leakage:** Features are purely socioeconomic/household columns.
# MAGIC Label is derived from immunization record columns — completely separate domain.
# MAGIC
# MAGIC **Clinical use case:** ANM visits a household with a newborn.
# MAGIC She knows the household situation but has no vaccination history yet.
# MAGIC This model tells her: *will this child likely dropout before completing the schedule?*

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

# MAGIC %md ## 2. Define Columns

# COMMAND ----------

# Features: ONLY household + maternal characteristics
# Zero overlap with immunization columns used in the label
CAT_COLS = [
    "social_group_code",
    "rural",
    "cooking_fuel",
    "toilet_used",
    "is_television",
    "is_telephone",
    "highest_qualification",   # mother's education
    "house_structure",
    "drinking_water_source",
    "where_del_took_place",    # institutional delivery → better vaccination start
    "source_of_anc"
]

NUM_COLS = [
    "age",                     # mother's age
    "w_preg_no",               # parity
    "no_of_anc",               # ANC visits — proxy for healthcare engagement
    "had_anc_registration"     # engineered binary flag
]

ALL_FEATURE_COLS = CAT_COLS + NUM_COLS

# Label columns (immunization record — separate domain from features)
IMMUN_COLS = ["bcg_vaccine", "no_of_polio_doses_ri",
              "no_of_dpt_injection", "measles", "ever_vacination_taken_bye_baby"]

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

# 3b. Filter to rows where immunization data is recorded
spark_df = spark_df.filter(
    F.col("ever_vacination_taken_bye_baby").isNotNull() &
    (F.col("ever_vacination_taken_bye_baby") != "NA")
)

# 3c. Convert to pandas
pdf = spark_df.select(ALL_FEATURE_COLS + IMMUN_COLS).toPandas()
print(f"Rows available for Model 3 training: {len(pdf)}")

# COMMAND ----------

# MAGIC %md ## 4. Build Label

# COMMAND ----------

# Fully immunized = ALL four conditions met:
#   BCG received (any time)
#   Polio doses >= 3
#   DPT injections >= 3
#   Measles received
#
# incomplete_immunization = 1 if any condition fails

pdf["no_of_polio_doses_ri"] = pd.to_numeric(
    pdf["no_of_polio_doses_ri"], errors="coerce"
).fillna(0)
pdf["no_of_dpt_injection"] = pd.to_numeric(
    pdf["no_of_dpt_injection"], errors="coerce"
).fillna(0)

# Treat value "9" in polio/DPT as unknown — exclude these rows
pdf = pdf[
    (pdf["no_of_polio_doses_ri"] != 9) &
    (pdf["no_of_dpt_injection"]  != 9)
]

bcg_received      = pdf["bcg_vaccine"].str.strip().str.startswith("Yes")
polio_complete    = pdf["no_of_polio_doses_ri"] >= 3
dpt_complete      = pdf["no_of_dpt_injection"]  >= 3
measles_received  = pdf["measles"].str.strip() == "Yes"

fully_immunized = bcg_received & polio_complete & dpt_complete & measles_received
pdf["incomplete_immunization"] = (~fully_immunized).astype(int)

complete_count   = fully_immunized.sum()
incomplete_count = len(pdf) - complete_count
print(f"Fully immunized   : {complete_count} ({100*complete_count/len(pdf):.1f}%)")
print(f"Incomplete/dropout: {incomplete_count} ({100*incomplete_count/len(pdf):.1f}%)")

# COMMAND ----------

# MAGIC %md ## 5. Encode Features

# COMMAND ----------

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
print(f"Feature count: {len(ENCODED_FEATURES)}")

# COMMAND ----------

# MAGIC %md ## 6. Train/Test Split

# COMMAND ----------

X = pdf[ENCODED_FEATURES].values
y = pdf["incomplete_immunization"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")
print(f"Train dropout rate: {y_train.mean()*100:.1f}%")
print(f"Test  dropout rate: {y_test.mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md ## 7. Train Model (Option 1: class_weight="balanced")

# COMMAND ----------

with mlflow.start_run(run_name="model3_immunization_dropout"):
    mlflow.log_params({
        "target":          "incomplete_immunization",
        "approach":        "household_features_only",
        "class_weight":    "balanced",
        "n_estimators":    100,
        "max_depth":       8,
        "train_size":      len(X_train),
        "test_size":       len(X_test),
        "dropout_rate_%":  round(y_train.mean() * 100, 1)
    })

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    proba  = clf.predict_proba(X_test)[:, 1]
    pred   = clf.predict(X_test)
    auc    = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    mlflow.log_metrics({
        "auc_roc":     auc,
        "pr_auc":      pr_auc,
        "sensitivity": sensitivity,
        "TP":          tp,
        "FN":          fn,
        "FP":          fp,
        "TN":          tn
    })

    sig = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(clf, "model_immunization_dropout", signature=sig)

    print(f"AUC-ROC     : {auc:.4f}")
    print(f"PR-AUC      : {pr_auc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}  (recall on dropout class)")
    print(f"TP: {tp}  FN: {fn}  FP: {fp}  TN: {tn}")

# COMMAND ----------

# MAGIC %md ## 8. Feature Importance

# COMMAND ----------

importances = pd.DataFrame({
    "feature":    ENCODED_FEATURES,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

print("Top features driving immunization dropout risk:")
spark.createDataFrame(importances).display()

# COMMAND ----------

# MAGIC %md ## 9. Save Risk Scores to Delta Lake

# COMMAND ----------

# Score all rows and write predictions back to Delta
pdf["dropout_risk_score"] = clf.predict_proba(
    pdf[ENCODED_FEATURES].values
)[:, 1]

pdf["risk_level"] = pd.cut(
    pdf["dropout_risk_score"],
    bins=[0, 0.4, 0.65, 1.0],
    labels=["Low", "Medium", "High"]
)

score_df = spark.createDataFrame(
    pdf[["dropout_risk_score", "risk_level", "incomplete_immunization"]]
)

score_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", True) \
    .saveAsTable("workspace.suraksha.immunization_scores")

print("Risk scores saved to workspace.suraksha.immunization_scores")
spark.sql("SELECT risk_level, COUNT(*) as count FROM workspace.suraksha.immunization_scores "
          "GROUP BY risk_level ORDER BY risk_level").show()