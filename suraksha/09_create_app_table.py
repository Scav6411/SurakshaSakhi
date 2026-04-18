# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Create App Patient Table
# MAGIC
# MAGIC Builds `workspace.suraksha.patients` — the single table used by the FastAPI
# MAGIC backend and React frontend.
# MAGIC
# MAGIC **Source:** `workspace.suraksha.raw_survey`  (filter: `is_currently_pregnant = 'Yes'`)
# MAGIC
# MAGIC **Columns included:**
# MAGIC - Identity + demographics (shared across all models)
# MAGIC - All feature columns from Models 1, 2, 3, 4
# MAGIC - All target columns (NULL — populated later by scoring pipeline)
# MAGIC - Risk score columns (NULL — populated by scoring pipeline)
# MAGIC - App management columns (last_visit_date, visit_count, is_deleted, …)
# MAGIC
# MAGIC Run this once to bootstrap the table, then run `07_score_pipeline` to fill
# MAGIC the risk scores.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, BooleanType, LongType
)
from datetime import datetime

# COMMAND ----------

# MAGIC %md ## 1. Load Currently Pregnant Women

# COMMAND ----------

raw = spark.table("workspace.suraksha.raw_survey")
print(f"Total rows in raw_survey: {raw.count()}")

pregnant = raw.filter(
    F.col("is_currently_pregnant").isin("Yes", "1", 1) |
    (F.col("is_currently_pregnant") == True)
)
print(f"Currently pregnant women: {pregnant.count()}")

# COMMAND ----------

# MAGIC %md ## 2. Engineer had_anc_registration flag

# COMMAND ----------

pregnant = pregnant.withColumn(
    "had_anc_registration",
    F.when(
        F.col("swelling_of_hand_feet_face").isNull() |
        (F.col("swelling_of_hand_feet_face") == "NA"),
        0
    ).otherwise(1).cast(IntegerType())
)

# COMMAND ----------

# MAGIC %md ## 3. Select All Columns
# MAGIC
# MAGIC ### Column map (by model)
# MAGIC
# MAGIC | Model | Features used |
# MAGIC |---|---|
# MAGIC | M1 Complications | demographics + ANC symptoms + household |
# MAGIC | M2 Home Delivery | demographics + household + marital + source_of_anc |
# MAGIC | M3 Immunization | household + maternal + delivery location |
# MAGIC | M4 Child Mortality | birth chars + complications + ANC symptoms + household + birth weight |

# COMMAND ----------

# ── Identity ────────────────────────────────────────────────────────────────
IDENTITY_COLS = [
    "w_id",           # survey respondent ID — becomes patient_id
    "PSU_ID",
]

# ── Demographics (all models) ────────────────────────────────────────────────
DEMO_COLS = [
    "age",
    "rural",
    "marital_status",
    "social_group_code",
    "highest_qualification",
    "w_preg_no",              # parity
    "mother_age_when_baby_was_born",  # Model 4
    "order_of_birth",                 # Model 4
]

# ── ANC / pregnancy symptom columns (Model 1 features, Model 4 features) ────
ANC_SYMPTOM_COLS = [
    "swelling_of_hand_feet_face",
    "hypertension_high_bp",
    "excessive_bleeding",
    "paleness_giddiness_weakness",
    "visual_disturbance",
    "excessive_vomiting",
    "convulsion_not_from_fever",
    "no_of_anc",
    "source_of_anc",
    "no_of_tt_injections",        # Model 4
    "consumption_of_ifa",         # Model 4
]

# ── Household columns (Models 1,2,3,4) ──────────────────────────────────────
HOUSEHOLD_COLS = [
    "cooking_fuel",
    "toilet_used",
    "is_telephone",
    "is_television",
    "house_structure",
    "drinking_water_source",
]

# ── Delivery / birth columns (Models 2,3,4) ─────────────────────────────────
DELIVERY_COLS = [
    "where_del_took_place",           # Model 2 TARGET / Model 3 & 4 feature
    "type_of_delivery",               # Model 4
    "type_of_birth",                  # Model 4
    "gender",                         # Model 4 (child gender)
    "who_conducted_del_at_home",      # Model 4
    "check_up_with_48_hours_of_del",  # Model 4
    "first_breast_feeding",           # Model 4
]

# ── Birth weight (Model 4 — 69% NA, Model A only) ───────────────────────────
WEIGHT_COLS = [
    "weight_of_baby_kg",
    "weight_of_baby_grams",
]

# ── Model 1 TARGETS — delivery complications (will be NULL for pregnant women)
M1_TARGET_COLS = [
    "premature_labour",
    "prolonged_labour",
    "obstructed_labour",
    "excessive_bleeding_during_birth",
    "convulsion_high_bp",
    "breech_presentation",
]

# ── Model 3 immunization columns (label derivation, NULL for pregnant women) ─
IMMUN_COLS = [
    "bcg_vaccine",
    "no_of_polio_doses_ri",
    "no_of_dpt_injection",
    "measles",
    "ever_vacination_taken_bye_baby",
]

# ── Model 4 TARGET ───────────────────────────────────────────────────────────
M4_TARGET_COLS = [
    "kind_of_birth",   # Live Birth Surviving / Not-Surviving / Still Birth
]

# COMMAND ----------

ALL_SURVEY_COLS = (
    IDENTITY_COLS +
    DEMO_COLS +
    ANC_SYMPTOM_COLS +
    HOUSEHOLD_COLS +
    DELIVERY_COLS +
    WEIGHT_COLS +
    M1_TARGET_COLS +
    IMMUN_COLS +
    M4_TARGET_COLS +
    ["had_anc_registration"]
)

# Only keep columns that actually exist in raw_survey
available = set(pregnant.columns)
selected  = [c for c in ALL_SURVEY_COLS if c in available]
missing   = [c for c in ALL_SURVEY_COLS if c not in available]

print(f"Selected columns : {len(selected)}")
print(f"Missing from raw : {missing}")

pdf_base = pregnant.select(selected)

# COMMAND ----------

# MAGIC %md ## 4. Add App Management + Risk Score Columns

# COMMAND ----------

now = datetime.utcnow().isoformat()

app_df = (
    pdf_base
    # Use w_id as the unique patient identifier
    .withColumnRenamed("w_id", "patient_id")

    # Display name — NULL initially (ANM fills in when registering)
    .withColumn("name",             F.lit(None).cast(StringType()))
    .withColumn("weeks_pregnant",   F.lit(None).cast(IntegerType()))
    .withColumn("notes",            F.lit(None).cast(StringType()))

    # ── Risk scores from Model 1 (complications) ──────────────────────
    .withColumn("risk_complication",  F.lit(None).cast(DoubleType()))

    # ── Risk score from Model 2 (home delivery) ───────────────────────
    .withColumn("risk_home_delivery", F.lit(None).cast(DoubleType()))

    # ── Risk score from Model 3 (immunization dropout) ────────────────
    .withColumn("risk_immunization",  F.lit(None).cast(DoubleType()))

    # ── Risk score from Model 4 (child mortality) ─────────────────────
    .withColumn("risk_child_mortality", F.lit(None).cast(DoubleType()))

    # ── Priority level (set by scoring pipeline) ──────────────────────
    .withColumn("priority_level",     F.lit(None).cast(StringType()))

    # ── App state ──────────────────────────────────────────────────────
    .withColumn("last_visit_date",  F.lit(None).cast(StringType()))
    .withColumn("visit_count",      F.lit(0).cast(IntegerType()))
    .withColumn("is_deleted",       F.lit(False).cast(BooleanType()))
    .withColumn("created_at",       F.lit(now).cast(StringType()))
)

print(f"Final column count : {len(app_df.columns)}")
print(f"Rows               : {app_df.count()}")
app_df.printSchema()

# COMMAND ----------

# MAGIC %md ## 5. Write to Delta Lake

# COMMAND ----------

app_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.suraksha.patients")

print("✓ Saved to workspace.suraksha.patients")

# COMMAND ----------

# MAGIC %md ## 6. Sanity Check

# COMMAND ----------

spark.sql("""
    SELECT
        COUNT(*)                                                    AS total,
        SUM(CASE WHEN priority_level IS NULL THEN 1 ELSE 0 END)    AS not_yet_scored,
        SUM(CASE WHEN name IS NULL           THEN 1 ELSE 0 END)    AS unnamed,
        COUNT(DISTINCT PSU_ID)                                      AS villages
    FROM workspace.suraksha.patients
""").show()

# COMMAND ----------

spark.sql("""
    SELECT patient_id, PSU_ID, age, rural, social_group_code,
           had_anc_registration,
           risk_complication, risk_home_delivery, risk_immunization,
           risk_child_mortality, priority_level
    FROM workspace.suraksha.patients
    ORDER BY patient_id
    LIMIT 10
""").display()

# COMMAND ----------

# MAGIC %md ## 7. Create visit_log table (if not exists)

# COMMAND ----------

spark.sql("""
    CREATE TABLE IF NOT EXISTS workspace.suraksha.visit_log (
        visit_id     STRING,
        patient_id   STRING,
        visit_date   STRING,
        notes        STRING,
        overall_risk DOUBLE
    ) USING DELTA
""")

print("✓ visit_log table ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Column Reference
# MAGIC
# MAGIC | Column group | Columns | Source |
# MAGIC |---|---|---|
# MAGIC | Identity | patient_id (=w_id), PSU_ID | raw_survey |
# MAGIC | Demographics | age, rural, marital_status, social_group_code, highest_qualification, w_preg_no | raw_survey |
# MAGIC | ANC symptoms | swelling_of_hand_feet_face, hypertension_high_bp, excessive_bleeding, paleness_giddiness_weakness, visual_disturbance, excessive_vomiting, convulsion_not_from_fever | raw_survey |
# MAGIC | ANC metrics | no_of_anc, source_of_anc, had_anc_registration | raw_survey / engineered |
# MAGIC | Household | cooking_fuel, toilet_used, is_telephone, is_television, house_structure, drinking_water_source | raw_survey |
# MAGIC | Delivery | where_del_took_place, type_of_delivery, type_of_birth, gender, who_conducted_del_at_home, check_up_with_48_hours_of_del, first_breast_feeding | raw_survey |
# MAGIC | M1 targets | premature_labour, prolonged_labour, obstructed_labour, excessive_bleeding_during_birth, convulsion_high_bp, breech_presentation | NULL → filled by models |
# MAGIC | M3 label cols | bcg_vaccine, no_of_polio_doses_ri, no_of_dpt_injection, measles, ever_vacination_taken_bye_baby | raw_survey |
# MAGIC | M4 target | kind_of_birth | NULL → filled by models |
# MAGIC | Risk scores | risk_complication, risk_home_delivery, risk_immunization, risk_child_mortality, overall_risk, risk_level | NULL → run 07_score_pipeline |
# MAGIC | App state | name, weeks_pregnant, notes, last_visit_date, visit_count, is_deleted, created_at | NULL / defaults |
