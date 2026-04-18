"""
Batch scoring endpoint.

Flow:
  1. Fetch patients from Delta Lake (all, or only unscored).
  2. Score each one in Python (preprocessing pipeline + RandomForest models).
  3. Push scores back in one MERGE statement per chunk of 200 rows.

POST /api/score/batch          — score only patients where risk_level IS NULL
POST /api/score/batch?all=true — rescore every non-deleted patient
POST /api/score/{patient_id}   — rescore a single patient
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Query

from ..database import run_query, execute, esc
from ..scoring import score_patient, get_loaded_models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/score", tags=["scoring"])


def _update_sql(r: dict) -> str:
    def _v(x) -> str:
        return "NULL" if x is None else str(x)

    pid = r["patient_id"].replace("'", "''")
    pl  = (r.get("priority_level") or "Medically Fit").replace("'", "''")
    return (
        f"UPDATE workspace.suraksha.patients SET"
        f"  risk_complication    = {_v(r['risk_complication'])},"
        f"  risk_home_delivery   = {_v(r['risk_home_delivery'])},"
        f"  risk_immunization    = {_v(r['risk_immunization'])},"
        f"  risk_child_mortality = {_v(r['risk_child_mortality'])},"
        f"  priority_level       = '{pl}'"
        f"  WHERE patient_id = '{pid}'"
    )


def _run_batch(score_all: bool = False) -> dict:
    """Core scoring logic — called synchronously or from a background task."""
    models_ready = get_loaded_models()
    if not models_ready:
        return {"status": "skipped", "reason": "no models loaded", "scored": 0}

    where = "WHERE is_deleted = false" if score_all else "WHERE is_deleted = false AND priority_level IS NULL"

    patients = run_query(f"SELECT * FROM workspace.suraksha.patients {where}")
    total    = len(patients)
    logger.info(f"Batch scoring: {total} patients (score_all={score_all})")

    if total == 0:
        return {"status": "ok", "scored": 0, "message": "No patients to score"}

    updated = 0
    errors  = 0
    # Score and update first patient with full detail, rest with normal logging
    for i, p in enumerate(patients):
        pid = p.get("patient_id")
        try:
            sc = score_patient(dict(p))
            sql = _update_sql({"patient_id": pid, **sc})
            if i == 0:
                logger.info(f"First patient SQL: {sql}")
            execute(sql)
            updated += 1
            if i == 0:
                logger.info(f"First patient update succeeded: pid={pid} scores={sc}")
        except Exception as exc:
            logger.warning(f"Scoring/update failed for {pid}: {exc}", exc_info=True)
            errors += 1
            if i == 0:
                logger.error(f"First patient FULL error for pid={pid}", exc_info=True)
        if updated % 50 == 0 and updated > 0:
            logger.info(f"Progress: {updated}/{total}")

    return {
        "status":  "ok",
        "total":   total,
        "scored":  updated,
        "errors":  errors,
        "models":  models_ready,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/batch")
async def batch_score(
    background_tasks: BackgroundTasks,
    all: Optional[bool] = Query(default=False, description="Rescore all patients, not just unscored ones"),
    background: Optional[bool] = Query(default=False, description="Run in background (returns immediately)"),
):
    """
    Trigger batch scoring.

    - Default: only patients where risk_level IS NULL (new / unscored rows).
    - ?all=true: rescore every non-deleted patient.
    - ?background=true: fire-and-forget, returns immediately.
    """
    if background:
        background_tasks.add_task(_run_batch, score_all=all)
        return {"status": "started", "mode": "background"}

    return _run_batch(score_all=all)


@router.post("/{patient_id}")
async def score_one(patient_id: str):
    """Rescore a single patient and persist the new scores."""
    rows = run_query(f"""
        SELECT * FROM workspace.suraksha.patients
        WHERE patient_id = {esc(patient_id)} AND is_deleted = false
    """)
    if not rows:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Patient not found")

    sc = score_patient(dict(rows[0]))

    cm = sc["risk_child_mortality"]
    execute(f"""
        UPDATE workspace.suraksha.patients SET
            risk_complication    = {sc['risk_complication']},
            risk_home_delivery   = {sc['risk_home_delivery']},
            risk_immunization    = {sc['risk_immunization']},
            risk_child_mortality = {'NULL' if cm is None else cm},
            priority_level       = {esc(sc.get('priority_level', 'Medically Fit'))}
        WHERE patient_id = {esc(patient_id)}
    """)

    return {"patient_id": patient_id, **sc}
