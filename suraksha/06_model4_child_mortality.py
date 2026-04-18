# Databricks notebook source
# MAGIC %md
# MAGIC # Suraksha — Model 4: Child Mortality Risk
# MAGIC
# MAGIC Predicts probability of child death (P(child dies)) using birth-time features.
# MAGIC
# MAGIC **Three sub-models:**
# MAGIC - **Model A**: With birth weight (strongest predictor, 69% NA — ~3,200 rows)
# MAGIC - **Model B**: Without birth weight (all ~10,000 rows — wider coverage)
# MAGIC - **Model C**: Model B + post-birth vaccine record ⚠️ leakage risk — see Section 2
# MAGIC
# MAGIC **Class imbalance strategy: SMOTE + class_weight="balanced"**
# MAGIC
# MAGIC **Label:** `kind_of_birth`
# MAGIC - `Live Birth Surviving`     → child_died = 0
# MAGIC - `Live Birth Not-Surviving` → child_died = 1
# MAGIC - `Still Birth`              → excluded
# MAGIC
# MAGIC **Evaluation:** PR-AUC (primary) + AUC-ROC

# COMMAND ----------

# MAGIC %pip install scikit-learn imbalanced-learn mlflow --quiet

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from mlflow.models.signature import infer_signature
from imblearn.over_sampling import SMOTE
from pyspark.sql import functions as F

print("Libraries loaded")

# COMMAND ----------

# MAGIC %md ## 1. Load from Delta Lake

# COMMAND ----------

try:
    spark_df = spark.table("workspace.suraksha.features")
    print("Loaded from workspace.suraksha.features")
except Exception:
    spark_df = spark.table("workspace.suraksha.raw_survey")
    print("Loaded from workspace.suraksha.raw_survey")

print(f"Total rows: {spark_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Definitions
# MAGIC
# MAGIC ### Temporal safety rules
# MAGIC | Group | When known | Safe to use? |
# MAGIC |---|---|---|
# MAGIC | Birth characteristics | At delivery | ✅ |
# MAGIC | Delivery complications | At delivery | ✅ |
# MAGIC | ANC symptoms | During pregnancy | ✅ |
# MAGIC | Socioeconomic / household | Before birth | ✅ |
# MAGIC | Birth weight | At delivery | ✅ (69% NA) |
# MAGIC | `bcg_vaccine` | Day 0–1 (given at birth) | ✅ proxy for birth-care quality |
# MAGIC | `no_of_polio_doses_ri`, `no_of_dpt_injection` | 6+ weeks post-birth | ⚠️ leakage — dead children have 0 doses |
# MAGIC | `measles` | 9+ months post-birth | ❌ strong leakage — survival selection |
# MAGIC | `is_suffering_from_ari`, `baby_suffer_from_fever`, `is_suffering_from_diarrhoea` | Post-birth morbidity | ❌ only recorded for living children |

# COMMAND ----------

# ── ANC symptom columns (58% NA — structural, means no ANC registration) ──
ANC_SYMPTOM_COLS = [
    "hypertension_high_bp",           # preeclampsia → fetal growth restriction
    "paleness_giddiness_weakness",    # maternal anaemia → low birth weight
    "swelling_of_hand_feet_face",     # preeclampsia indicator
    "excessive_bleeding",             # antepartum haemorrhage → fetal distress
    "convulsion_not_from_fever",      # eclampsia → high neonatal mortality
]

# ── All 6 delivery complication columns ───────────────────────────────────
COMPLICATION_COLS = [
    "premature_labour",               # already proven predictor
    "prolonged_labour",               # → birth asphyxia
    "obstructed_labour",              # leading cause of birth asphyxia deaths
    "excessive_bleeding_during_birth",# maternal haemorrhage → fetal distress
    "convulsion_high_bp",             # eclampsia at delivery
    "breech_presentation",            # malpresentation → trauma / asphyxia
]

# ── Base categorical features (birth-time & socioeconomic) ────────────────
CAT_COLS_BASE = [
    # Birth characteristics
    "type_of_birth",                  # single / multiple
    "gender",
    "where_del_took_place",           # home vs institution
    "type_of_delivery",               # normal / C-section / instrumental
    "who_conducted_del_at_home",      # trained / untrained attendant
    "check_up_with_48_hours_of_del",  # immediate postnatal check
    "first_breast_feeding",           # colostrum — protective
    # Maternal care
    "consumption_of_ifa",             # IFA during pregnancy
    "source_of_anc",                  # govt / private / ASHA / none
    "marital_status",                 # support system proxy
    # Socioeconomic
    "social_group_code",
    "rural",
    "highest_qualification",          # mother's education
    "cooking_fuel",
    "toilet_used",
    "is_television",                  # wealth proxy
    "is_telephone",                   # emergency access
    "house_structure",                # pucca / semi-pucca / kutcha
    "drinking_water_source",          # neonatal infection risk
] + COMPLICATION_COLS + ANC_SYMPTOM_COLS

# ── Base numerical features ───────────────────────────────────────────────
NUM_COLS_BASE = [
    "order_of_birth",                 # birth order
    "mother_age_when_baby_was_born",
    "no_of_anc",
    "no_of_tt_injections",
    "w_preg_no",                      # parity — grand multiparity risk
    "had_anc_registration",           # engineered flag (see Section 3)
]

# ── Birth weight (69% NA — Model A only) ─────────────────────────────────
WEIGHT_COLS = ["weight_of_baby_kg", "weight_of_baby_grams"]

# ── Vaccine columns ───────────────────────────────────────────────────────
# BCG: given at birth → safe proxy for whether child received immediate care
BCG_COL = ["bcg_vaccine"]

# Post-birth vaccines: ⚠️ DATA LEAKAGE RISK
# Children who died young have 0 doses purely because they didn't survive
# long enough to receive them — not because the vaccine caused death.
# Model C intentionally includes these to quantify the leakage effect.
# DO NOT use Model C scores as the production model.
POSTNATAL_VACCINE_COLS = [
    "no_of_polio_doses_ri",           # ⚠️ given 6+ weeks post-birth
    "no_of_dpt_injection",            # ⚠️ given 6+ weeks post-birth
    "measles",                        # ❌ given 9+ months — strong survival selection
]

LABEL_COL = "kind_of_birth"

# COMMAND ----------

# MAGIC %md ## 3. Preprocessing

# COMMAND ----------

# 3a. Filter: keep only live births (exclude still births)
spark_df = spark_df.filter(
    F.col(LABEL_COL).isin("Live Birth Surviving", "Live Birth Not-Surviving")
)

# 3b. Engineer had_anc_registration flag
#     Structural NA in ANC symptom cols = woman never registered for ANC
spark_df = spark_df.withColumn(
    "had_anc_registration",
    F.when(
        F.col("hypertension_high_bp").isNull() |
        (F.col("hypertension_high_bp") == "NA"), 0.0
    ).otherwise(1.0)
)

# 3c. Fill ANC symptom NAs with "Not_Reported" (not random — means no ANC)
for c in ANC_SYMPTOM_COLS:
    spark_df = spark_df.withColumn(
        c,
        F.when(F.col(c).isNull() | (F.col(c) == "NA"), "Not_Reported")
         .otherwise(F.col(c))
    )

# 3d. Fill complication NAs with "No" (not recorded = did not occur)
for c in COMPLICATION_COLS:
    spark_df = spark_df.withColumn(
        c,
        F.when(F.col(c).isNull() | (F.col(c) == "NA"), "No")
         .otherwise(F.col(c))
    )

# 3e. Pull all needed columns to pandas
all_cols = list(set(
    CAT_COLS_BASE + NUM_COLS_BASE + WEIGHT_COLS +
    BCG_COL + POSTNATAL_VACCINE_COLS + [LABEL_COL]
))
available = set(spark_df.columns)
all_cols  = [c for c in all_cols if c in available]

pdf = spark_df.select(all_cols).toPandas()
print(f"Rows after filtering still births: {len(pdf)}")

# COMMAND ----------

# 3f. Binary label
pdf["child_died"] = (pdf[LABEL_COL] == "Live Birth Not-Surviving").astype(int)

died  = pdf["child_died"].sum()
alive = len(pdf) - died
print(f"\nClass distribution:")
print(f"  Survived : {alive} ({100*alive/len(pdf):.1f}%)")
print(f"  Died     : {died}  ({100*died/len(pdf):.1f}%)")

# COMMAND ----------

# 3g. Encode categoricals
encoders = {}

for c in CAT_COLS_BASE:
    pdf[c] = pdf[c].fillna("Unknown").astype(str)
    le = LabelEncoder()
    pdf[c + "_enc"] = le.fit_transform(pdf[c])
    encoders[c] = le

# BCG
for c in BCG_COL:
    if c in pdf.columns:
        pdf[c] = pdf[c].fillna("Unknown").astype(str)
        le = LabelEncoder()
        pdf[c + "_enc"] = le.fit_transform(pdf[c])
        encoders[c] = le

# Post-birth vaccines — numeric doses + measles binary
for c in ["no_of_polio_doses_ri", "no_of_dpt_injection"]:
    if c in pdf.columns:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)

if "measles" in pdf.columns:
    pdf["measles"] = pdf["measles"].fillna("Unknown").astype(str)
    le = LabelEncoder()
    pdf["measles_enc"] = le.fit_transform(pdf["measles"])
    encoders["measles"] = le

# Numerics
for c in NUM_COLS_BASE:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)

for c in WEIGHT_COLS:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

# ── Build encoded feature lists ───────────────────────────────────────────
ENCODED_BASE = [c + "_enc" for c in CAT_COLS_BASE] + NUM_COLS_BASE
ENCODED_BASE_BCG = ENCODED_BASE + ["bcg_vaccine_enc"] if "bcg_vaccine" in pdf.columns else ENCODED_BASE

ENCODED_WITH_WT  = ENCODED_BASE_BCG + WEIGHT_COLS

POSTNATAL_EXTRA  = []
for c in ["no_of_polio_doses_ri", "no_of_dpt_injection"]:
    if c in pdf.columns:
        POSTNATAL_EXTRA.append(c)
if "measles" in pdf.columns:
    POSTNATAL_EXTRA.append("measles_enc")
ENCODED_WITH_VACCINES = ENCODED_BASE_BCG + POSTNATAL_EXTRA

print(f"Base features (Model B)           : {len(ENCODED_BASE)}")
print(f"Base + BCG (Models A/B)           : {len(ENCODED_BASE_BCG)}")
print(f"Base + BCG + weight (Model A)     : {len(ENCODED_WITH_WT)}")
print(f"Base + BCG + vaccines (Model C ⚠️): {len(ENCODED_WITH_VACCINES)}")

# COMMAND ----------

# MAGIC %md ## 4. Split into Sub-Models

# COMMAND ----------

# Model A — rows where birth weight is recorded
pdf_A = pdf.dropna(subset=WEIGHT_COLS).copy()
print(f"Model A rows (with birth weight): {len(pdf_A)}")
print(f"  Deaths: {pdf_A['child_died'].sum()} ({100*pdf_A['child_died'].mean():.1f}%)")

# Model B — all rows, no weight columns
pdf_B = pdf.copy()
print(f"\nModel B rows (all data): {len(pdf_B)}")
print(f"  Deaths: {pdf_B['child_died'].sum()} ({100*pdf_B['child_died'].mean():.1f}%)")

# Model C — all rows, post-birth vaccines included (leakage benchmark)
pdf_C = pdf.copy()
print(f"\nModel C rows (⚠️ includes post-birth vaccines): {len(pdf_C)}")

# COMMAND ----------

# MAGIC %md ## 5. Train Helper

# COMMAND ----------

def train_mortality_model(X_train, X_test, y_train, y_test,
                          model_name, feature_names, use_lr=False):
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))

    if use_lr:
        clf   = LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=42, solver="lbfgs")
        algo  = "LogisticRegression"
        scaler = StandardScaler()
    else:
        clf   = RandomForestClassifier(n_estimators=100, max_depth=8,
                                       class_weight="balanced",
                                       random_state=42, n_jobs=-1)
        algo  = "RandomForest"
        scaler = None

    with mlflow.start_run(run_name=f"{model_name}__{algo}", nested=True):
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"\n  [{algo}] After SMOTE: {y_res.sum()} deaths / "
              f"{len(y_res)} total ({100*y_res.mean():.1f}%)")

        if scaler:
            X_res    = scaler.fit_transform(X_res)
            X_test_s = scaler.transform(X_test)
        else:
            X_test_s = X_test

        clf.fit(X_res, y_res)
        proba  = clf.predict_proba(X_test_s)[:, 1]
        pred   = clf.predict(X_test_s)
        auc    = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0

        mlflow.log_params({
            "model": model_name, "algorithm": algo,
            "smote": True, "class_weight": "balanced",
            "features_count": len(feature_names)
        })
        mlflow.log_metrics({
            "auc_roc": auc, "pr_auc": pr_auc,
            "sensitivity": sensitivity, "precision": precision,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn
        })
        sig = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(clf, f"model_{model_name}_{algo.lower()}",
                                 signature=sig)

        print(f"  [{algo}] PR-AUC: {pr_auc:.4f}  AUC-ROC: {auc:.4f}  "
              f"Sensitivity: {sensitivity:.4f}  Precision: {precision:.4f}")
        print(f"  TP: {tp}  FN: {fn}  FP: {fp}  TN: {tn}")

    return auc, pr_auc, clf

# COMMAND ----------

# MAGIC %md ## 6. Train All Sub-Models

# COMMAND ----------

results = []

with mlflow.start_run(run_name="model4_child_mortality"):

    # ── Model A: With birth weight ────────────────────────────────────────
    print("=" * 60)
    print("MODEL A — With Birth Weight (~3,200 rows)")
    print("=" * 60)

    X_A = pdf_A[ENCODED_WITH_WT].values
    y_A = pdf_A["child_died"].values
    X_A_tr, X_A_te, y_A_tr, y_A_te = train_test_split(
        X_A, y_A, test_size=0.2, random_state=42, stratify=y_A
    )
    for use_lr in [True, False]:
        auc, pr, clf = train_mortality_model(
            X_A_tr, X_A_te, y_A_tr, y_A_te,
            "modelA_with_weight", ENCODED_WITH_WT, use_lr=use_lr
        )
        algo = "LR" if use_lr else "RF"
        results.append({"model": "A_with_weight", "algo": algo,
                        "auc_roc": round(auc, 4), "pr_auc": round(pr, 4)})
        if not use_lr:
            clf_A_rf = clf

    # ── Model B: Base features + BCG ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL B — Base + BCG, No Birth Weight (~10,000 rows)")
    print("=" * 60)

    X_B = pdf_B[ENCODED_BASE_BCG].values
    y_B = pdf_B["child_died"].values
    X_B_tr, X_B_te, y_B_tr, y_B_te = train_test_split(
        X_B, y_B, test_size=0.2, random_state=42, stratify=y_B
    )
    for use_lr in [True, False]:
        auc, pr, clf = train_mortality_model(
            X_B_tr, X_B_te, y_B_tr, y_B_te,
            "modelB_no_weight", ENCODED_BASE_BCG, use_lr=use_lr
        )
        algo = "LR" if use_lr else "RF"
        results.append({"model": "B_no_weight", "algo": algo,
                        "auc_roc": round(auc, 4), "pr_auc": round(pr, 4)})
        if not use_lr:
            clf_B_rf = clf

    # ── Model C: With post-birth vaccines ⚠️ ─────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL C — ⚠️ Post-birth vaccines included (leakage benchmark)")
    print("Expected inflated PR-AUC — DO NOT use for production scoring")
    print("=" * 60)

    X_C = pdf_C[ENCODED_WITH_VACCINES].values
    y_C = pdf_C["child_died"].values
    X_C_tr, X_C_te, y_C_tr, y_C_te = train_test_split(
        X_C, y_C, test_size=0.2, random_state=42, stratify=y_C
    )
    auc, pr, clf_C_rf = train_mortality_model(
        X_C_tr, X_C_te, y_C_tr, y_C_te,
        "modelC_with_vaccines", ENCODED_WITH_VACCINES, use_lr=False
    )
    results.append({"model": "C_vaccines_⚠️", "algo": "RF",
                    "auc_roc": round(auc, 4), "pr_auc": round(pr, 4)})

# COMMAND ----------

# MAGIC %md ## 7. Feature Importance

# COMMAND ----------

importances_B = pd.DataFrame({
    "feature":    ENCODED_BASE_BCG,
    "importance": clf_B_rf.feature_importances_
}).sort_values("importance", ascending=False)

print("Model B — Top features driving child mortality risk:")
spark.createDataFrame(importances_B).display()

# COMMAND ----------

importances_A = pd.DataFrame({
    "feature":    ENCODED_WITH_WT,
    "importance": clf_A_rf.feature_importances_
}).sort_values("importance", ascending=False)

print("Model A — Top features (with birth weight):")
spark.createDataFrame(importances_A).display()

# COMMAND ----------

# MAGIC %md ## 8. Comparison Summary

# COMMAND ----------

results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)

print("\n" + "=" * 65)
print("MODEL 4 — CHILD MORTALITY COMPARISON")
print("Primary metric: PR-AUC  |  Base rate ≈ 5%  |  Random = 0.05")
print("Model C PR-AUC will be inflated — it's a leakage sanity check")
print("=" * 65)
print(results_df.to_string(index=False))

spark.createDataFrame(results_df).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Interpreting Model C (Leakage Check)
# MAGIC
# MAGIC If Model C's PR-AUC is **significantly higher** than Model B, it confirms
# MAGIC that post-birth vaccine columns are acting as survival proxies, not causal
# MAGIC predictors. The gap `PR-AUC(C) - PR-AUC(B)` is the leakage magnitude.
# MAGIC
# MAGIC **Use Model B RF for production scoring** — it has the most rows, no
# MAGIC leakage, and adds BCG as a safe birth-time care-quality indicator.

# COMMAND ----------

# MAGIC %md ## 10. Score and Save to Delta Lake

# COMMAND ----------

pdf["mortality_risk_score"] = np.nan

mask_A = pdf[WEIGHT_COLS].notna().all(axis=1)
if mask_A.sum() > 0:
    pdf.loc[mask_A, "mortality_risk_score"] = (
        clf_A_rf.predict_proba(pdf.loc[mask_A, ENCODED_WITH_WT].values)[:, 1]
    )

mask_B = ~mask_A
if mask_B.sum() > 0:
    pdf.loc[mask_B, "mortality_risk_score"] = (
        clf_B_rf.predict_proba(pdf.loc[mask_B, ENCODED_BASE_BCG].values)[:, 1]
    )

pdf["model_used"] = np.where(mask_A, "ModelA_with_weight", "ModelB_no_weight")
pdf["risk_level"]  = pd.cut(
    pdf["mortality_risk_score"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

score_df = spark.createDataFrame(
    pdf[["mortality_risk_score", "model_used",
         "risk_level", "child_died"]].astype(str)
)
score_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", True) \
    .saveAsTable("workspace.suraksha.child_mortality_scores")

print("Scores saved to workspace.suraksha.child_mortality_scores")

spark.sql("""
    SELECT model_used, risk_level, COUNT(*) as count
    FROM workspace.suraksha.child_mortality_scores
    GROUP BY model_used, risk_level
    ORDER BY model_used, risk_level
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. New Feature Summary
# MAGIC
# MAGIC | Group | Columns added | Rationale |
# MAGIC |---|---|---|
# MAGIC | Delivery complications | `prolonged_labour`, `obstructed_labour`, `excessive_bleeding_during_birth`, `convulsion_high_bp`, `breech_presentation` | Same temporal position as `premature_labour` — all known at delivery |
# MAGIC | ANC symptoms | `hypertension_high_bp`, `paleness_giddiness_weakness`, `swelling_of_hand_feet_face`, `excessive_bleeding`, `convulsion_not_from_fever` | Maternal health during pregnancy directly affects fetal outcomes |
# MAGIC | Socioeconomic | `is_television`, `is_telephone`, `house_structure`, `drinking_water_source` | Wealth/sanitation proxies not previously in Model 4 |
# MAGIC | Maternal | `source_of_anc`, `marital_status`, `w_preg_no`, `had_anc_registration` | ANC quality, parity, support system |
# MAGIC | Vaccines — safe | `bcg_vaccine` | Given at birth — proxy for whether child received immediate care |
# MAGIC | Vaccines — ⚠️ | `no_of_polio_doses_ri`, `no_of_dpt_injection`, `measles` | Post-birth survival selection — Model C only, not production |
