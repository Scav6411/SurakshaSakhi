"""
Scoring — loads models from MLflow experiments, calls preprocessing.preprocess(),
runs inference, returns risk scores.

Flow per patient:
  patient dict
    └─► preprocessing.preprocess(cat_cols, num_cols, encoders) → feature matrix
        └─► sklearn RandomForest.predict_proba()               → probabilities
            └─► score_patient()                                → risk dict
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import (
    preprocess,
    ANC_SYMPTOM_COLS, M4_COMPLICATION_COLS,
    M1_CAT_COLS, M1_NUM_COLS,
    M2_CAT_COLS, M2_NUM_COLS,
    M3_CAT_COLS, M3_NUM_COLS,
    M4_CAT_COLS, M4_NUM_COLS,
)

logger = logging.getLogger(__name__)

# ── Pinned MLflow run IDs ─────────────────────────────────────────────────────
M1_RUNS: dict[str, tuple[str, str]] = {
    "premature_labour":                ("96c95d21939640dc98522a32097cdc0c", "model_premature_labour_opt1"),
    "prolonged_labour":                ("6655dc63a2ec4367951690aebe0d9881", "model_prolonged_labour_opt1"),
    "obstructed_labour":               ("de373bbc36b74b6e989c9fe3eb35ef64", "model_obstructed_labour_opt1"),
    "excessive_bleeding_during_birth": ("7265630886d14380a6b2d65ab608f744", "model_excessive_bleeding_during_birth_opt1"),
    "convulsion_high_bp":              ("4ee88bcfbaaa488a8b8b1a2693aed6d0", "model_convulsion_high_bp_opt1"),
    "breech_presentation":             ("2b64a90857e6480fa7f8a7bc7ea4b5d9", "model_breech_presentation_opt1"),
}
M2_RUN: tuple[str, str] = ("ab8c0f0e863f4a40b62f2c988b433a24", "model_home_delivery_opt1")
M3_RUN: tuple[str, str] = ("a2cae0b331e34aa5b2d57f9d5e915777", "model_immunization_dropout")
M4_RUN: tuple[str, str] = ("fe2e8c2051ec40a7ab97884242a91f9d", "model_modelB_no_weight_randomforest")

# Bundle format:
#   "complications"   → {"clfs": {target: clf}, "encoders", "cat_cols", "num_cols"}
#   others            → {"clf": clf,             "encoders", "cat_cols", "num_cols"}
_models: dict[str, Any] = {}


# ── Encoder fitting ───────────────────────────────────────────────────────────

def _fetch_training_df() -> pd.DataFrame:
    """
    Pull completed-pregnancy rows from raw_survey and run through the same
    preprocessing pipeline so fitted LabelEncoders match training.
    """
    from .database import run_query
    from .preprocessing import add_anc_flag, fill_anc_symptoms, fill_complication_cols

    all_cat = list(set(M1_CAT_COLS + M2_CAT_COLS + M3_CAT_COLS + M4_CAT_COLS))
    extra   = ["swelling_of_hand_feet_face"] + ANC_SYMPTOM_COLS + M4_COMPLICATION_COLS
    cols    = list(set(all_cat + extra))

    rows = run_query(f"""
        SELECT {', '.join(cols)}
        FROM workspace.suraksha.patients
    """)

    if not rows:
        logger.warning("No training rows returned — encoders may be incomplete")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = add_anc_flag(df)
    df = fill_anc_symptoms(df)
    df = fill_complication_cols(df)
    if "where_del_took_place" in df.columns:
        df["where_del_took_place"] = df["where_del_took_place"].fillna("Unknown")

    return df


def _fit_encoders(cat_cols: list, df: pd.DataFrame) -> dict:
    """Fit one LabelEncoder per categorical column. Always includes sentinel values."""
    from sklearn.preprocessing import LabelEncoder

    encoders: dict = {}
    for col in cat_cols:
        vals = df[col].fillna("Unknown").astype(str).tolist() if col in df.columns else []
        vals += ["Unknown", "Not_Reported", "No", "Yes"]
        le = LabelEncoder()
        le.fit(sorted(set(vals)))
        encoders[col] = le
    return encoders


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> None:
    """
    App startup:
    1. Set MLflow tracking URI → Databricks workspace.
    2. Fetch raw_survey training data → fit LabelEncoders via preprocessing pipeline.
    3. Load each sklearn model from its pinned MLflow run artifact.
    """
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri("databricks")

    logger.info("Fitting LabelEncoders from training data…")
    try:
        df = _fetch_training_df()
        logger.info(f"Training rows fetched: {len(df)}")
    except Exception as exc:
        logger.error(f"Training data fetch failed: {exc}")
        df = pd.DataFrame()

    enc_m1 = _fit_encoders(M1_CAT_COLS, df)
    enc_m2 = _fit_encoders(M2_CAT_COLS, df)
    enc_m3 = _fit_encoders(M3_CAT_COLS, df)
    enc_m4 = _fit_encoders(M4_CAT_COLS, df)

    def _load(run_id: str, artifact: str):
        return mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact}")

    # Model 1 — 6 complication classifiers
    logger.info("Loading Model 1 (complications)…")
    clfs_m1: dict = {}
    for target, (run_id, artifact) in M1_RUNS.items():
        try:
            clfs_m1[target] = _load(run_id, artifact)
            logger.info(f"  ✓ {target}")
        except Exception as exc:
            logger.warning(f"  ✗ {target}: {exc}")
    if clfs_m1:
        _models["complications"] = {
            "clfs": clfs_m1, "encoders": enc_m1,
            "cat_cols": M1_CAT_COLS, "num_cols": M1_NUM_COLS,
        }

    # Model 2 — home delivery
    logger.info("Loading Model 2 (home delivery)…")
    try:
        _models["home_delivery"] = {
            "clf": _load(*M2_RUN), "encoders": enc_m2,
            "cat_cols": M2_CAT_COLS, "num_cols": M2_NUM_COLS,
        }
        logger.info("  ✓ home_delivery")
    except Exception as exc:
        logger.warning(f"  ✗ M2: {exc}")

    # Model 3 — immunization dropout
    logger.info("Loading Model 3 (immunization)…")
    try:
        _models["immunization"] = {
            "clf": _load(*M3_RUN), "encoders": enc_m3,
            "cat_cols": M3_CAT_COLS, "num_cols": M3_NUM_COLS,
        }
        logger.info("  ✓ immunization")
    except Exception as exc:
        logger.warning(f"  ✗ M3: {exc}")

    # Model 4 — child mortality (Model B RF)
    logger.info("Loading Model 4 (child mortality)…")
    try:
        _models["child_mortality"] = {
            "clf": _load(*M4_RUN), "encoders": enc_m4,
            "cat_cols": M4_CAT_COLS, "num_cols": M4_NUM_COLS,
        }
        logger.info("  ✓ child_mortality (Model B RF)")
    except Exception as exc:
        logger.warning(f"  ✗ M4: {exc}")

    logger.info(f"Models ready: {list(_models.keys())}")


def get_loaded_models() -> list[str]:
    return list(_models.keys())


# ── Inference ─────────────────────────────────────────────────────────────────

def score_patient(patient: dict) -> dict:
    """
    patient dict
      └─► preprocessing.preprocess()  (engineering + encoding)
          └─► model.predict_proba()
              └─► risk scores dict

    M1: preprocess once → run all 6 classifiers → take max probability.
    M2, M3, M4: preprocess → single classifier.
    """
    pid = patient.get("patient_id", "?")
    logger.info(f"[{pid}] score_patient called, models available: {list(_models.keys())}")

    scores: dict[str, Any] = {
        "risk_complication":    0.0,
        "risk_home_delivery":   0.0,
        "risk_immunization":    0.0,
        "risk_child_mortality": None,
    }

    # ── Model 1: preprocess → 6 classifiers → max proba ──────────────────
    comp = _models.get("complications")
    if comp:
        try:
            X = preprocess(patient, comp["cat_cols"], comp["num_cols"], comp["encoders"])
            logger.info(f"[{pid}] M1 input shape: {X.shape}")
            comp_probas = np.stack(
                [clf.predict_proba(X)[:, 1] for clf in comp["clfs"].values()],
                axis=1
            )
            scores["risk_complication"] = round(float(comp_probas.max(axis=1)[0]), 4)
            logger.info(f"[{pid}] M1 risk_complication={scores['risk_complication']}")
        except Exception as exc:
            logger.warning(f"[{pid}] M1 error: {exc}", exc_info=True)
    else:
        logger.warning(f"[{pid}] M1 not loaded")

    # ── Model 2: preprocess → single classifier ───────────────────────────
    hd = _models.get("home_delivery")
    if hd:
        try:
            X = preprocess(patient, hd["cat_cols"], hd["num_cols"], hd["encoders"])
            logger.info(f"[{pid}] M2 input shape: {X.shape}")
            scores["risk_home_delivery"] = round(float(hd["clf"].predict_proba(X)[:, 1][0]), 4)
            logger.info(f"[{pid}] M2 risk_home_delivery={scores['risk_home_delivery']}")
        except Exception as exc:
            logger.warning(f"[{pid}] M2 error: {exc}", exc_info=True)
    else:
        logger.warning(f"[{pid}] M2 not loaded")

    # ── Model 3: preprocess → single classifier ───────────────────────────
    im = _models.get("immunization")
    if im:
        try:
            X = preprocess(patient, im["cat_cols"], im["num_cols"], im["encoders"])
            logger.info(f"[{pid}] M3 input shape: {X.shape}")
            scores["risk_immunization"] = round(float(im["clf"].predict_proba(X)[:, 1][0]), 4)
            logger.info(f"[{pid}] M3 risk_immunization={scores['risk_immunization']}")
        except Exception as exc:
            logger.warning(f"[{pid}] M3 error: {exc}", exc_info=True)
    else:
        logger.warning(f"[{pid}] M3 not loaded")

    # ── Model 4: preprocess → single classifier ───────────────────────────
    cm = _models.get("child_mortality")
    if cm:
        try:
            X = preprocess(patient, cm["cat_cols"], cm["num_cols"], cm["encoders"])
            logger.info(f"[{pid}] M4 input shape: {X.shape}")
            scores["risk_child_mortality"] = round(float(cm["clf"].predict_proba(X)[:, 1][0]), 4)
            logger.info(f"[{pid}] M4 risk_child_mortality={scores['risk_child_mortality']}")
        except Exception as exc:
            logger.warning(f"[{pid}] M4 error: {exc}", exc_info=True)
    else:
        logger.warning(f"[{pid}] M4 not loaded")

    # ── Priority level — mirrors 07_score_pipeline.py Section 6 ──────────
    # Cut on risk_complication alone (same bins/labels as 07)
    comp_risk = scores["risk_complication"]
    scores["priority_level"] = (
        "High Priority" if comp_risk >= 0.6 else
        "Routine Care"  if comp_risk >= 0.3 else
        "Medically Fit"
    )

    return scores
