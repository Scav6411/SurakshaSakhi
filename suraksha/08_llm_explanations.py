# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — LLM Risk Explanations
# MAGIC
# MAGIC For each **High-risk** pregnant woman, calls `databricks-meta-llama-3-3-70b-instruct`
# MAGIC to generate a plain-English explanation of *why* she is critical and *what the ANM
# MAGIC should prioritise* on her next visit.
# MAGIC
# MAGIC **Input:**  `workspace.suraksha.risk_scores` + `workspace.suraksha.raw_survey`
# MAGIC **Output:** `workspace.suraksha.patient_explanations`
# MAGIC **Columns:** `w_id`, `PSU_ID`, `overall_risk`, `risk_level`, `explanation`, `generated_at`

# COMMAND ----------

import mlflow.deployments
import pandas as pd
import time
from datetime import datetime
from pyspark.sql import functions as F

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
deploy_client = mlflow.deployments.get_deploy_client("databricks")
print(f"LLM endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md ## 1. Load Critical Patients

# COMMAND ----------

# Load risk scores and join with raw survey for feature details
scores = spark.table("workspace.suraksha.risk_scores")
survey = spark.table("workspace.suraksha.raw_survey")

# Feature columns we want to pull for the explanation prompt
DETAIL_COLS = [
    "w_id",
    # Maternal
    "age", "marital_status", "highest_qualification",
    "w_preg_no",                        # parity
    "rural", "social_group_code", "PSU_ID",
    # ANC
    "no_of_anc", "source_of_anc",
    "swelling_of_hand_feet_face", "hypertension_high_bp",
    "excessive_bleeding", "paleness_giddiness_weakness",
    "visual_disturbance", "excessive_vomiting", "convulsion_not_from_fever",
    # Household
    "cooking_fuel", "toilet_used", "house_structure",
    "drinking_water_source", "is_telephone",
    # Delivery complications (for post-birth review)
    "premature_labour", "prolonged_labour", "obstructed_labour",
    "excessive_bleeding_during_birth", "convulsion_high_bp", "breech_presentation",
]

available = set(survey.columns)
detail_cols = [c for c in DETAIL_COLS if c in available]

# Keep only High-risk patients
high_risk = scores.filter(F.col("risk_level") == "High")
print(f"High-risk patients: {high_risk.count()}")

# Join with raw survey on w_id
joined = high_risk.join(
    survey.select(detail_cols),
    on="w_id", how="left"
)

pdf = joined.toPandas()
print(f"Rows with full details: {len(pdf)}")
pdf.head(2)

# COMMAND ----------

# MAGIC %md ## 2. Prompt Builder

# COMMAND ----------

SYSTEM_PROMPT = """You are a maternal health assistant helping Auxiliary Nurse Midwives (ANMs)
in rural Bihar, India prioritise their home visits.
Given a patient's profile and risk scores, write a clear 3-sentence explanation:
1. What are the top 2-3 specific risk factors making her critical.
2. What is the most likely adverse outcome if she is not visited soon.
3. What the ANM should check or do on the next visit.
Be specific, factual, and avoid generic advice. Write in plain English."""


def safe(row, col, default="Unknown"):
    v = row.get(col)
    return str(v) if pd.notna(v) and str(v) not in ("nan", "NA", "") else default


def yn(row, col):
    """Return 'Yes' / 'No' / 'Not reported' for symptom columns."""
    v = safe(row, col, "Not_Reported")
    return v if v in ("Yes", "No") else "Not reported"


def build_prompt(row: dict) -> str:
    # Symptom flags
    symptoms = []
    sym_map = {
        "hypertension_high_bp":          "Hypertension / High BP",
        "swelling_of_hand_feet_face":    "Swelling of hands/feet/face",
        "excessive_bleeding":            "Excessive bleeding (antepartum)",
        "paleness_giddiness_weakness":   "Paleness / anaemia symptoms",
        "visual_disturbance":            "Visual disturbance",
        "excessive_vomiting":            "Excessive vomiting",
        "convulsion_not_from_fever":     "Convulsions (non-fever)",
    }
    for col, label in sym_map.items():
        if yn(row, col) == "Yes":
            symptoms.append(label)

    # Delivery complications
    complications = []
    comp_map = {
        "premature_labour":                "Premature labour",
        "prolonged_labour":                "Prolonged labour",
        "obstructed_labour":               "Obstructed labour",
        "excessive_bleeding_during_birth": "Excessive bleeding during birth",
        "convulsion_high_bp":              "Convulsions / high BP at delivery",
        "breech_presentation":             "Breech presentation",
    }
    for col, label in comp_map.items():
        if safe(row, col, "No") == "Yes":
            complications.append(label)

    anc_visits   = safe(row, "no_of_anc", "0")
    anc_source   = safe(row, "source_of_anc", "None")
    parity       = safe(row, "w_preg_no", "Unknown")
    age          = safe(row, "age", "Unknown")
    education    = safe(row, "highest_qualification", "Unknown")
    social_group = safe(row, "social_group_code", "Unknown")
    toilet       = safe(row, "toilet_used", "Unknown")
    cooking      = safe(row, "cooking_fuel", "Unknown")
    house        = safe(row, "house_structure", "Unknown")
    water        = safe(row, "drinking_water_source", "Unknown")
    telephone    = safe(row, "is_telephone", "No")
    psu          = safe(row, "PSU_ID", "Unknown")
    rural        = safe(row, "rural", "Unknown")

    r_comp  = float(row.get("risk_complication",  0) or 0)
    r_home  = float(row.get("risk_home_delivery", 0) or 0)
    r_immun = float(row.get("risk_immunization",  0) or 0)
    r_ovr   = float(row.get("overall_risk",       0) or 0)

    prompt = f"""Patient Profile — PSU {psu} ({rural})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maternal:
  Age: {age} | Pregnancy #: {parity} | Education: {education} | Social group: {social_group}

Antenatal Care:
  ANC visits: {anc_visits} | ANC source: {anc_source}
  Symptoms reported during pregnancy: {', '.join(symptoms) if symptoms else 'None reported'}

Delivery:
  Complications: {', '.join(complications) if complications else 'None recorded'}

Household:
  Toilet: {toilet} | Cooking fuel: {cooking}
  House: {house} | Water source: {water} | Telephone: {telephone}

Risk Scores (0–100%):
  Delivery complication : {r_comp*100:.0f}%
  Home delivery         : {r_home*100:.0f}%
  Immunization dropout  : {r_immun*100:.0f}%
  Overall (weighted)    : {r_ovr*100:.0f}%  ← HIGH RISK

Task: Explain in exactly 3 sentences why this woman is at high risk and what the ANM must do."""

    return prompt

# COMMAND ----------

# Test prompt for first patient
sample = pdf.iloc[0].to_dict()
print(build_prompt(sample))

# COMMAND ----------

# MAGIC %md ## 3. Generate Explanations

# COMMAND ----------

def call_llm(prompt: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            response = deploy_client.predict(
                endpoint=LLM_ENDPOINT,
                inputs={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "max_tokens": 250,
                    "temperature": 0.3,
                },
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                return f"[LLM error: {e}]"


explanations = []
total = len(pdf)

for i, row in pdf.iterrows():
    prompt = build_prompt(row.to_dict())
    explanation = call_llm(prompt)

    explanations.append({
        "w_id":         str(row.get("w_id", "")),
        "PSU_ID":       str(row.get("PSU_ID", "")),
        "overall_risk": float(row.get("overall_risk", 0) or 0),
        "risk_level":   str(row.get("risk_level", "High")),
        "explanation":  explanation,
        "generated_at": datetime.utcnow().isoformat(),
    })

    idx = len(explanations)
    print(f"[{idx}/{total}] PSU {row.get('PSU_ID')} | risk={float(row.get('overall_risk',0))*100:.0f}%")
    print(f"  → {explanation[:120]}…\n")

    time.sleep(0.5)  # stay within rate limits

print(f"\nDone — {len(explanations)} explanations generated")

# COMMAND ----------

# MAGIC %md ## 4. Save to Delta Lake

# COMMAND ----------

exp_pdf = pd.DataFrame(explanations)
exp_sdf = spark.createDataFrame(exp_pdf)

exp_sdf.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", True) \
    .saveAsTable("workspace.suraksha.patient_explanations")

print("Saved to workspace.suraksha.patient_explanations")

spark.sql("""
    SELECT PSU_ID,
           ROUND(overall_risk * 100, 1) AS risk_pct,
           explanation
    FROM workspace.suraksha.patient_explanations
    ORDER BY overall_risk DESC
    LIMIT 10
""").show(truncate=80)

# COMMAND ----------

# MAGIC %md ## 5. Preview — Top 5 Most Critical

# COMMAND ----------

spark.sql("""
    SELECT PSU_ID,
           ROUND(overall_risk * 100, 1) AS risk_pct,
           explanation
    FROM workspace.suraksha.patient_explanations
    ORDER BY overall_risk DESC
    LIMIT 5
""").display()
