# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Notebook 01: Data Ingestion
# MAGIC Reads the AHS Bihar Araria CSV from Unity Catalog volume and writes it as a Delta Lake table.

# COMMAND ----------

# Create dedicated schema for the project
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.suraksha")
print("Schema workspace.suraksha ready")

# COMMAND ----------

# Read CSV from Unity Catalog volume
raw_df = (
    spark.read
    .option("header", True)
    .option("encoding", "latin-1")
    .option("inferSchema", True)
    .csv("/Volumes/workspace/default/suraksha_data/ahs_araria.csv")
)

print(f"Rows   : {raw_df.count()}")
print(f"Columns: {len(raw_df.columns)}")

# COMMAND ----------

# Write as Delta table — raw, no transformations
(
    raw_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.suraksha.raw_survey")
)

print("Delta table workspace.suraksha.raw_survey written")

# COMMAND ----------

# Verify
spark.sql("SELECT COUNT(*) AS total_rows FROM workspace.suraksha.raw_survey").show()

# COMMAND ----------

# Quick sanity check on key columns
spark.sql("""
    SELECT
        COUNT(*)                                          AS total,
        SUM(CASE WHEN is_currently_pregnant = 'Yes'
            THEN 1 ELSE 0 END)                           AS currently_pregnant,
        SUM(CASE WHEN is_currently_pregnant = 'No'
            THEN 1 ELSE 0 END)                           AS completed_pregnancies,
        COUNT(DISTINCT PSU_ID)                            AS total_psu_villages
    FROM workspace.suraksha.raw_survey
""").show()

# COMMAND ----------

# Show column nullability summary for our 3 model feature sets
from pyspark.sql import functions as F

df = spark.table("workspace.suraksha.raw_survey")
total = df.count()

key_cols = [
    "age", "w_preg_no", "rural", "marital_status", "social_group_code",
    "no_of_anc", "swelling_of_hand_feet_face", "hypertension_high_bp",
    "where_del_took_place", "toilet_used", "cooking_fuel", "is_telephone",
    "bcg_vaccine", "no_of_polio_doses_ri", "measles", "immunization_card"
]

print(f"{'Column':<40} {'NA %':>8}")
print("-" * 50)
for col in key_cols:
    if col in df.columns:
        na_count = df.filter(
            (F.col(col).isNull()) | (F.col(col) == "NA")
        ).count()
        pct = (na_count / total) * 100
        flag = "🔴" if pct > 50 else ("🟡" if pct > 20 else "🟢")
        print(f"{flag} {col:<38} {pct:>7.1f}%")
