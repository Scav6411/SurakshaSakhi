# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Score Pipeline
# MAGIC
# MAGIC Scores the **currently pregnant women** using pretrained models for complications,
# MAGIC home delivery, and immunization, ranks them by overall risk, and writes results to Delta Lake.
# MAGIC
# MAGIC **Output table:** `workspace.suraksha.risk_scores`
# MAGIC Columns: `w_id`, `PSU_ID`, `age`, `rural`, `social_group_code`,
# MAGIC `risk_complication`, `risk_home_delivery`, `risk_immunization`,
# MAGIC `priority_level`, `priority_rank`

# COMMAND ----------

import joblib
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window

MODEL_DIR = "/Volumes/workspace/default/suraksha_data/models"
print(f"Reading models from: {MODEL_DIR}")

# COMMAND ----------

# MAGIC %md ## 1. Load Pretrained Models

# COMMAND ----------

print("Loading Complications Model...")
m1_models = joblib.load(f"{MODEL_DIR}/model1_complications_rf.pkl")
# Use the config from the first complication's model to get shared encoders/cols
m1_sample_target = list(m1_models.keys())[0]
M1_CAT_COLS = m1_models[m1_sample_target]["cat_cols"]
M1_NUM_COLS = m1_models[m1_sample_target]["num_cols"]
m1_encoders = m1_models[m1_sample_target]["encoders"]

print("Loading Home Delivery Model...")
m2_bundle = joblib.load(f"{MODEL_DIR}/model2_home_delivery_rf.pkl")
clf_m2 = m2_bundle["model"]
M2_CAT_COLS = m2_bundle["cat_cols"]
M2_NUM_COLS = m2_bundle["num_cols"]
m2_encoders = m2_bundle["encoders"]

print("Loading Immunization Model...")
m3_bundle = joblib.load(f"{MODEL_DIR}/model3_immunization_rf.pkl")
clf_m3 = m3_bundle["model"]
M3_CAT_COLS = m3_bundle["cat_cols"]
M3_NUM_COLS = m3_bundle["num_cols"]
m3_encoders = m3_bundle["encoders"]

# COMMAND ----------

# MAGIC %md ## 2. Load Raw Data

# COMMAND ----------

raw = spark.table("workspace.suraksha.raw_survey")
score_spark  = raw.filter(F.col("is_currently_pregnant") == "Yes")
print(f"Scoring rows (currently pregnant): {score_spark.count()}")

# COMMAND ----------

# MAGIC %md ## 3. Shared Preprocessing Helper

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

def add_anc_flag(sdf):
    """Engineer had_anc_registration flag (1 = registered, 0 = not)."""
    return sdf.withColumn(
        "had_anc_registration",
        F.when(
            F.col("swelling_of_hand_feet_face").isNull() |
            (F.col("swelling_of_hand_feet_face") == "NA"), 0.0
        ).otherwise(1.0)
    )

def fill_anc_symptoms(sdf):
    """Fill NULL ANC symptom columns with Not_Reported."""
    for c in ANC_SYMPTOM_COLS:
        sdf = sdf.withColumn(
            c,
            F.when(F.col(c).isNull() | (F.col(c) == "NA"), "Not_Reported")
             .otherwise(F.col(c))
        )
    return sdf

def encode_pdf(pdf, cat_cols, num_cols, encoders=None):
    """
    Encode a pandas DataFrame using provided encoders (for scoring new data).
    Returns (encoded_X, feature_names)
    """
    if encoders is None:
        encoders = {}

    for c in num_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)

    for c in cat_cols:
        pdf[c] = pdf[c].fillna("Unknown").astype(str)
        le = encoders.get(c)
        if le is not None:
            # Handle unseen categories
            known = set(le.classes_)
            pdf[c + "_safe"] = pdf[c].apply(lambda v: v if v in known else le.classes_[0])
            pdf[c + "_enc"] = le.transform(pdf[c + "_safe"])
        else:
            pdf[c + "_enc"] = 0 # Fallback

    feature_names = [c + "_enc" for c in cat_cols] + num_cols
    X = pdf[feature_names].values
    return X, feature_names

# COMMAND ----------

# MAGIC %md ## 4. Prepare Currently Pregnant Women for Scoring

# COMMAND ----------

# All columns needed across 3 models
SCORE_COLS = list(set(
    ["w_id", "PSU_ID", "age", "rural", "social_group_code",
     "is_currently_pregnant"] +
    M1_CAT_COLS + M1_NUM_COLS +
    M2_CAT_COLS + M2_NUM_COLS +
    M3_CAT_COLS + M3_NUM_COLS +
    ANC_SYMPTOM_COLS +
    ["swelling_of_hand_feet_face"]   # needed for had_anc_registration
))

# Keep only columns that exist in the table
available = set(score_spark.columns)
SCORE_COLS = [c for c in SCORE_COLS if c in available]

score_spark_prep = add_anc_flag(fill_anc_symptoms(score_spark))
spdf = score_spark_prep.select(SCORE_COLS).toPandas()
print(f"Pregnant women to score: {len(spdf)}")

# For Model 3, where_del_took_place is unknown for currently pregnant → fill with 'Unknown'
if "where_del_took_place" in spdf.columns:
    spdf["where_del_took_place"] = spdf["where_del_took_place"].fillna("Unknown")

# COMMAND ----------

# MAGIC %md ## 5. Score Pipeline

# COMMAND ----------

# --- Model 1: Complications ---
if 'had_anc_registration' not in spdf.columns:
    spdf['had_anc_registration'] = spdf['swelling_of_hand_feet_face'].apply(
        lambda x: 0.0 if pd.isna(x) or x == 'NA' else 1.0
    )

X_s1, _ = encode_pdf(spdf.copy(), M1_CAT_COLS, M1_NUM_COLS, encoders=m1_encoders)

# Max probability across 6 complication classifiers
comp_probas = np.stack([
    cfg["model"].predict_proba(X_s1)[:, 1]
    for cfg in m1_models.values()
], axis=1)

spdf["risk_complication"] = comp_probas.max(axis=1).round(4)

# --- Model 2: Home Delivery ---
X_s2, _ = encode_pdf(spdf.copy(), M2_CAT_COLS, M2_NUM_COLS, encoders=m2_encoders)
spdf["risk_home_delivery"] = clf_m2.predict_proba(X_s2)[:, 1].round(4)

# --- Model 3: Immunization ---
X_s3, _ = encode_pdf(spdf.copy(), M3_CAT_COLS, M3_NUM_COLS, encoders=m3_encoders)
spdf["risk_immunization"] = clf_m3.predict_proba(X_s3)[:, 1].round(4)

# COMMAND ----------

# MAGIC %md ## 6. Compute Priority Level & Rank

# COMMAND ----------

# Assign priority level based on complications
spdf["priority_level"] = pd.cut(
    spdf["risk_complication"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Medically Fit", "Routine Care", "High Priority"],
    include_lowest=True
)

# Convert back to Spark SDF to apply distributed ranking
out_pdf = spdf.copy()
out_pdf["priority_level"] = out_pdf["priority_level"].astype(str)

out_sdf = spark.createDataFrame(out_pdf)

# Rank dynamically by complication -> home delivery -> immunization
windowSpec = Window.orderBy(
    F.col("risk_complication").desc(), 
    F.col("risk_home_delivery").desc(), 
    F.col("risk_immunization").desc()
)
out_sdf = out_sdf.withColumn("priority_rank", F.row_number().over(windowSpec))

# COMMAND ----------

# MAGIC %md ## 7. Save to Delta Lake

# COMMAND ----------

OUTPUT_COLS = [
    "w_id", "PSU_ID", "age", "rural", "social_group_code",
    "risk_complication", "risk_home_delivery", "risk_immunization",
    "priority_level", "priority_rank"
]

# Keep only output cols that exist
available_out = set(out_sdf.columns)
final_out_cols = [c for c in OUTPUT_COLS if c in available_out]

final_sdf = out_sdf.select(final_out_cols)

final_sdf.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", True) \
    .saveAsTable("workspace.suraksha.risk_scores")

print(f"Saved {final_sdf.count()} rows to workspace.suraksha.risk_scores")

spark.sql("""
    SELECT priority_level, COUNT(*) AS count,
           ROUND(AVG(risk_complication), 3) AS avg_complication_risk
    FROM workspace.suraksha.risk_scores
    GROUP BY priority_level
    ORDER BY avg_complication_risk DESC
""").show()

# COMMAND ----------

# MAGIC %md ## 8. Top 20 Priority Women

# COMMAND ----------

spark.sql("""
    SELECT priority_rank, w_id, PSU_ID, age, rural,
           ROUND(risk_complication * 100, 1)  AS complication_pct,
           ROUND(risk_home_delivery * 100, 1) AS home_del_pct,
           ROUND(risk_immunization  * 100, 1) AS immunization_pct,
           priority_level
    FROM workspace.suraksha.risk_scores
    ORDER BY priority_rank ASC
    LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 9. PSU (Village) Summary

# COMMAND ----------

spark.sql("""
    SELECT
        PSU_ID,
        COUNT(*)                                                               AS total_women,
        SUM(CASE WHEN priority_level = 'High Priority' THEN 1 ELSE 0 END)      AS high_priority,
        SUM(CASE WHEN priority_level = 'Routine Care'  THEN 1 ELSE 0 END)      AS routine_care,
        SUM(CASE WHEN priority_level = 'Medically Fit' THEN 1 ELSE 0 END)      AS medically_fit,
        ROUND(AVG(risk_complication), 3)                                       AS avg_complication_risk
    FROM workspace.suraksha.risk_scores
    GROUP BY PSU_ID
    ORDER BY high_priority DESC, avg_complication_risk DESC
""").show(30)