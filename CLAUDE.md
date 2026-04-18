# Suraksha — Project Context for Claude

## What This Project Is
**Suraksha** is a Maternal Risk Intelligence platform for ANMs (Auxiliary Nurse Midwives)
built for BharatBricks Hacks 2026 at IIT Indore (April 17–18, 2026).

It predicts which pregnant women in Araria district, Bihar are at highest risk —
so ANMs know who to visit first instead of going by geography or last contact.

**Hackathon track:** Swatantra (Open / Any Indic AI Use Case)
**Submission deadline:** 5:00 PM, April 18, 2026. Code freeze at 4:00 PM.

---

## Databricks Workspace

- **Profile name in ~/.databrickscfg:** `Ayush`
- **Host:** `https://dbc-bc7909d8-379d.cloud.databricks.com`
- **Primary user:** `me220003016@iiti.ac.in`
- **Team members added:**
  - `me220003080@iiti.ac.in`
  - `me220003085@iiti.ac.in`
  - `ee220002006@iiti.ac.in`
- **Unity Catalog:** `workspace.default` (only catalog available on Free Edition)
- **Volume:** `workspace.default.suraksha_data`
- **Schema for project tables:** `workspace.suraksha`
- **Always use `--profile Ayush`** with all databricks CLI commands

### Free Edition Constraints (critical)
- CPU-only, no GPU, ~15GB RAM
- **Public DBFS root is disabled** — use `/Volumes/workspace/default/suraksha_data/` instead
- **Spark ML classes are security-whitelisted** — `StringIndexer`, `VectorAssembler`, `Pipeline` from `pyspark.ml` throw `Py4JSecurityException`. Use sklearn instead.
- **MLflow Model Registry (S3) is blocked** — `registered_model_name` in `mlflow.sklearn.log_model()` throws `AccessDenied`. Remove it.
- **`mlflow.set_experiment()` causes GRPC error** — remove it, Databricks auto-creates experiment per notebook
- Models must run quantized on CPU — use Param-1 (2.9B), Sarvam-m, IndicTrans2

---

## Dataset

- **File:** `/Volumes/workspace/default/suraksha_data/ahs_araria.csv`
- **Delta table:** `workspace.suraksha.raw_survey`
- **Source:** Annual Health Survey — Women's Schedule, District Araria, Bihar
- **Size:** 10,606 rows × 350 columns, encoding: `latin-1`
- **1,734 currently pregnant women** (prediction targets)
- **8,872 completed pregnancies** (training data)

### Key NA Patterns
- `age`, `rural`, `marital_status`, `w_preg_no`, `social_group_code`: **0% NA** — solid base
- ANC symptom columns (`swelling_of_hand_feet_face`, `hypertension_high_bp`, etc.): **58.3% NA** — structural, means "no ANC registration", NOT random
- `transport_fac`, `asha_facilitator`: **59–79% NA** — outcome-correlated (only known if woman went to facility), **must be dropped as features**
- `reason_for_non_vaccination`, `months_of_preg_first_anc`: **96–97% NA** — drop entirely
- Immunization columns (`bcg_vaccine`, `no_of_polio_doses_ri`, `measles`): **11–12% NA** — manageable

---

## Three ML Models

### Model 1: Delivery Complication Risk (6 classifiers)
- **6 separate binary classifiers**, one per complication type
- **Targets:** `premature_labour`, `prolonged_labour`, `obstructed_labour`, `excessive_bleeding_during_birth`, `convulsion_high_bp`, `breech_presentation`
- **Features:** demographics + ANC-time symptoms (observed BEFORE delivery)
- **Not features:** the same symptom columns are NOT used as labels — temporal separation preserved
- **Label logic:** 1 = complication occurred at delivery, 0 = did not
- **Class balance:** 5–14% positive rate — use `class_weight="balanced"`
- **Notebook:** `/Workspace/suraksha/03_model1_complications`

### Model 2: Home Delivery Risk
- **Single binary classifier**
- **Target:** `where_del_took_place` → 1 = At Home, 0 = Institution
- **Features:** socioeconomic + demographic ONLY
- **Dropped features:** `transport_fac`, `asha_facilitator`, `financial_asistance_delivery` (all outcome-correlated)
- **Class balance:** 58% home / 42% institution — relatively balanced
- **Key metric:** Sensitivity (FN = missed home delivery is more dangerous than FP)
- **Notebook:** `/Workspace/suraksha/04_model2_home_delivery`

### Model 3: Immunization Dropout Risk
- **Single binary classifier**
- **Target:** `incomplete_immunization` (engineered from BCG + polio + DPT + measles)
- **Features:** ONLY household/socioeconomic columns — zero overlap with immunization columns
- **Label:** NOT(BCG received AND polio>=3 AND dpt>=3 AND measles=="Yes")
- **Exclude rows** where `no_of_polio_doses_ri` or `no_of_dpt_injection` == 9 (unknown code)
- **Approach:** Option 1 only (class_weight="balanced") — no comparison
- **Output Delta table:** `workspace.suraksha.immunization_scores`
- **Notebook:** `/Workspace/suraksha/05_model3_immunization`

---

## Key Design Decisions

1. **sklearn over Spark MLlib** — Spark ML is security-whitelisted on Free Edition. All models use sklearn RandomForest. Data is converted to pandas via `.toPandas()` (safe — Model 1 is ~4,418 rows, Model 2 ~10,175 rows).

2. **No `registered_model_name`** — Unity Catalog Model Registry S3 upload is blocked. Models logged as run artifacts only, accessible via MLflow UI.

3. **No `mlflow.set_experiment()`** — causes GRPC error on Free Edition. Databricks auto-links experiment to notebook.

4. **Temporal separation enforced across all models:**
   - Model 1: ANC symptoms (pregnancy) → predict delivery complications
   - Model 2: Household factors → predict delivery location
   - Model 3: Household factors → predict immunization completion
   - Label columns are NEVER reused as features

5. **`had_anc_registration`** — engineered binary flag. If `swelling_of_hand_feet_face` is NA, the woman never registered for ANC (structural NA). This is itself a risk factor.

6. **Class imbalance strategy:** `class_weight="balanced"` in RandomForest. AUC-ROC as evaluation metric (not accuracy — accuracy is misleading with 5–14% positive rates).

---

## Workspace Structure

```
/Workspace/suraksha/
├── 01_ingest.py                  ← CSV → Delta Lake raw_survey
├── 03_model1_complications.py    ← 6 complication classifiers
├── 04_model2_home_delivery.py    ← home delivery risk
└── 05_model3_immunization.py     ← immunization dropout

/Volumes/workspace/default/suraksha_data/
└── ahs_araria.csv

Delta Tables (workspace.suraksha):
├── raw_survey                    ← raw CSV
├── immunization_scores           ← Model 3 risk scores (written by notebook 05)
└── risk_scores                   ← TO BE CREATED by scoring pipeline   
```

---

## Remaining Work

1. **`06_score_pipeline.py`** — Apply best models to 1,734 currently pregnant women, write to `workspace.suraksha.risk_scores` with columns: `w_id`, `PSU_ID`, `age`, `risk_complication`, `risk_home_delivery`, `risk_immunization`, `overall_risk`, `risk_level`

2. **Streamlit App** — 4 tabs:
   - Tab 1: PSU bar chart (villages ranked by high-risk count)
   - Tab 2: Women table sorted by `overall_risk` with colour-coded badges
   - Tab 3: Individual risk card (3 scores + top 3 feature importances + recommended actions)
   - Tab 4: Hindi chatbot — RAG on ASHA guidelines + Param-1 (optional, build last)

3. **GitHub repo + README** — mandatory submission requirement (architecture diagram, run commands, demo steps)

4. **Demo video** — 2 minutes max

---

## Judging Criteria (from participant guide)

| Criterion | Weight | How we score |
|---|---|---|
| Databricks Usage | 30% | Delta Lake + PySpark + MLflow (18 runs) + Databricks App + Unity Catalog volume |
| Accuracy & Effectiveness | 25% | AUC-ROC logged in MLflow, feature importance shown |
| Innovation | 25% | District-specific trained model, prioritisation not just chatbot, Hindi RAG |
| Presentation & Demo | 20% | 4 clear screens, each tells a story in ~1 min |

---

## Local Project Path
`/home/ayush/bharatbricks/suraksha/`
