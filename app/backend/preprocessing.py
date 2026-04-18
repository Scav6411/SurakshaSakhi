"""
Preprocessing pipeline — mirrors 07_score_pipeline.py exactly.

Responsibilities:
  1. Column definitions for each model
  2. Feature engineering  (add_anc_flag, fill_anc_symptoms, fill_complication_cols)
  3. Encoding             (encode_pdf — pandas translation of 07_score_pipeline.py)
  4. preprocess()         — single entry point used by scoring.py
"""

import numpy as np
import pandas as pd

# ── Column definitions ────────────────────────────────────────────────────────

ANC_SYMPTOM_COLS = [
    "swelling_of_hand_feet_face",
    "hypertension_high_bp",
    "excessive_bleeding",
    "paleness_giddiness_weakness",
    "visual_disturbance",
    "excessive_vomiting",
    "convulsion_not_from_fever",
]

M4_COMPLICATION_COLS = [
    "premature_labour",
    "prolonged_labour",
    "obstructed_labour",
    "excessive_bleeding_during_birth",
    "convulsion_high_bp",
    "breech_presentation",
]

# Model 1: Delivery Complication Risk
M1_CAT_COLS = [
    "social_group_code", "rural", "toilet_used", "cooking_fuel",
    "is_telephone", "is_television", "highest_qualification", "source_of_anc",
] + ANC_SYMPTOM_COLS
M1_NUM_COLS = ["age", "w_preg_no", "no_of_anc", "had_anc_registration"]

# Model 2: Home Delivery Risk
M2_CAT_COLS = [
    "social_group_code", "rural", "marital_status", "toilet_used",
    "cooking_fuel", "is_telephone", "is_television", "highest_qualification",
    "source_of_anc", "house_structure", "drinking_water_source",
]
M2_NUM_COLS = ["age", "w_preg_no", "no_of_anc", "had_anc_registration"]

# Model 3: Immunization Dropout Risk
M3_CAT_COLS = [
    "social_group_code", "rural", "cooking_fuel", "toilet_used",
    "is_television", "is_telephone", "highest_qualification",
    "house_structure", "drinking_water_source",
    "where_del_took_place",   # 'Unknown' for currently pregnant women
    "source_of_anc",
]
M3_NUM_COLS = ["age", "w_preg_no", "no_of_anc", "had_anc_registration"]

# Model 4: Child Mortality Risk (Model B — no birth weight)
_M4_ANC_SUBSET = [
    "hypertension_high_bp", "paleness_giddiness_weakness",
    "swelling_of_hand_feet_face", "excessive_bleeding", "convulsion_not_from_fever",
]
M4_CAT_COLS = [
    "type_of_birth", "gender", "where_del_took_place", "type_of_delivery",
    "who_conducted_del_at_home", "check_up_with_48_hours_of_del",
    "first_breast_feeding", "consumption_of_ifa", "source_of_anc",
    "marital_status", "social_group_code", "rural", "highest_qualification",
    "cooking_fuel", "toilet_used", "is_television", "is_telephone",
    "house_structure", "drinking_water_source",
] + M4_COMPLICATION_COLS + _M4_ANC_SUBSET
M4_NUM_COLS = [
    "age", "order_of_birth", "mother_age_when_baby_was_born",
    "no_of_anc", "no_of_tt_injections", "w_preg_no", "had_anc_registration",
]


# ── Step 1: Feature engineering ───────────────────────────────────────────────

def add_anc_flag(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Derive had_anc_registration from swelling_of_hand_feet_face.
    Structural NA means the woman never registered for ANC.
    Mirrors add_anc_flag() in 07_score_pipeline.py.
    """
    pdf = pdf.copy()
    pdf["had_anc_registration"] = np.where(
        pdf["swelling_of_hand_feet_face"].isna() |
        (pdf["swelling_of_hand_feet_face"] == "NA"),
        0.0, 1.0
    )
    return pdf


def fill_anc_symptoms(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NULL ANC symptom columns with 'Not_Reported'.
    Structural missingness — not random.
    Mirrors fill_anc_symptoms() in 07_score_pipeline.py.
    """
    pdf = pdf.copy()
    for c in ANC_SYMPTOM_COLS:
        if c in pdf.columns:
            pdf[c] = pdf[c].where(pdf[c].notna() & (pdf[c] != "NA"), "Not_Reported")
    return pdf


def fill_complication_cols(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NULL delivery-complication columns with 'No'.
    Not recorded = did not occur.
    """
    pdf = pdf.copy()
    for c in M4_COMPLICATION_COLS:
        if c in pdf.columns:
            pdf[c] = pdf[c].where(pdf[c].notna() & (pdf[c] != "NA"), "No")
    return pdf


# ── Step 2: Encoding ──────────────────────────────────────────────────────────

def encode_pdf(pdf: pd.DataFrame, cat_cols: list, num_cols: list, encoders: dict) -> np.ndarray:
    """
    Encode a preprocessed DataFrame into a feature matrix for inference.

    Direct pandas translation of encode_pdf(fit_encoders=False) from 07_score_pipeline.py.
    Feature order: [cat_col_enc, ...] + [num_col, ...]   ← must match training.
    Unseen categories fall back to the first known class (safe, no crash).
    """
    pdf = pdf.copy()

    for c in num_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)

    for c in cat_cols:
        pdf[c] = pdf[c].fillna("Unknown").astype(str)
        le = encoders.get(c)
        if le is not None:
            known = set(le.classes_)
            pdf[c + "_safe"] = pdf[c].apply(lambda v: v if v in known else le.classes_[0])
            pdf[c + "_enc"]  = le.transform(pdf[c + "_safe"])
        else:
            pdf[c + "_enc"] = 0

    feature_cols = [c + "_enc" for c in cat_cols] + num_cols
    return pdf[feature_cols].values


# ── Step 3: Combined pipeline entry point ─────────────────────────────────────

def preprocess(patient: dict, cat_cols: list, num_cols: list, encoders: dict) -> np.ndarray:
    """
    Full preprocessing pipeline for a single patient dict:
      patient dict
        → DataFrame
        → add_anc_flag
        → fill_anc_symptoms
        → fill_complication_cols
        → fill where_del_took_place
        → encode_pdf
        → feature matrix (1 × n_features)

    This is the single entry point scoring.py calls for each model.
    """
    # Wrap in DataFrame so pandas operations work correctly
    pdf = pd.DataFrame([patient])

    # Ensure columns used in engineering steps exist
    for col in ANC_SYMPTOM_COLS + M4_COMPLICATION_COLS + ["swelling_of_hand_feet_face", "where_del_took_place"]:
        if col not in pdf.columns:
            pdf[col] = None

    # Engineering (order matters — same as 07_score_pipeline.py)
    pdf = add_anc_flag(pdf)
    pdf = fill_anc_symptoms(pdf)
    pdf = fill_complication_cols(pdf)

    # where_del_took_place is unknown for currently pregnant women (mirrors 07 line 151)
    pdf["where_del_took_place"] = pdf["where_del_took_place"].fillna("Unknown")

    # Encode and return feature matrix
    return encode_pdf(pdf, cat_cols, num_cols, encoders)
