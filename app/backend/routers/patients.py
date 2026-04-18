import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..database import run_query, execute, esc
from ..schemas import PatientIn
from ..scoring import score_patient

router = APIRouter(prefix="/api/patients", tags=["patients"])

# All user-editable columns in order — drives both INSERT and UPDATE.
# Must match the workspace.suraksha.patients schema from 09_create_app_table.py
_EDITABLE_COLS: list[tuple[str, str]] = [
    # fmt: (column_name, type)  type: "str" | "int" | "float" | "num"
    ("name",                            "str"),
    ("PSU_ID",                          "str"),
    ("age",                             "num"),
    ("weeks_pregnant",                  "num"),
    ("rural",                           "str"),
    ("marital_status",                  "str"),
    ("social_group_code",               "str"),
    ("highest_qualification",           "str"),
    ("w_preg_no",                       "num"),
    ("mother_age_when_baby_was_born",   "num"),
    ("order_of_birth",                  "num"),
    # ANC
    ("no_of_anc",                       "num"),
    ("source_of_anc",                   "str"),
    ("had_anc_registration",            "num"),
    ("no_of_tt_injections",             "num"),
    ("consumption_of_ifa",              "str"),
    ("swelling_of_hand_feet_face",      "str"),
    ("hypertension_high_bp",            "str"),
    ("excessive_bleeding",              "str"),
    ("paleness_giddiness_weakness",     "str"),
    ("visual_disturbance",              "str"),
    ("excessive_vomiting",              "str"),
    ("convulsion_not_from_fever",       "str"),
    # Household
    ("cooking_fuel",                    "str"),
    ("toilet_used",                     "str"),
    ("is_telephone",                    "str"),
    ("is_television",                   "str"),
    ("house_structure",                 "str"),
    ("drinking_water_source",           "str"),
    # Delivery / birth
    ("where_del_took_place",            "str"),
    ("type_of_delivery",                "str"),
    ("type_of_birth",                   "str"),
    ("gender",                          "str"),
    ("who_conducted_del_at_home",       "str"),
    ("check_up_with_48_hours_of_del",   "str"),
    ("first_breast_feeding",            "str"),
    ("weight_of_baby_kg",               "num"),
    ("weight_of_baby_grams",            "num"),
    # M1 targets
    ("premature_labour",                "str"),
    ("prolonged_labour",                "str"),
    ("obstructed_labour",               "str"),
    ("excessive_bleeding_during_birth", "str"),
    ("convulsion_high_bp",              "str"),
    ("breech_presentation",             "str"),
    # M3 label cols
    ("bcg_vaccine",                     "str"),
    ("no_of_polio_doses_ri",            "num"),
    ("no_of_dpt_injection",             "num"),
    ("measles",                         "str"),
    ("ever_vacination_taken_bye_baby",  "str"),
    # M4 target
    ("kind_of_birth",                   "str"),
]


def _sql_val(v, typ: str) -> str:
    """Convert a Python value to a SQL literal."""
    if v is None:
        return "NULL"
    if typ == "str":
        return esc(v)
    return str(v)  # numeric — emit as-is


def _build_insert(pid: str, body: PatientIn, sc: dict, now: str) -> str:
    data = body.model_dump()

    cols = ["patient_id"]
    vals = [esc(pid)]

    for col, typ in _EDITABLE_COLS:
        cols.append(col)
        vals.append(_sql_val(data.get(col), typ))

    # Risk scores (three model outputs + supplementary child mortality)
    for col in ("risk_complication", "risk_home_delivery",
                "risk_immunization", "risk_child_mortality"):
        cols.append(col)
        vals.append(str(sc.get(col)) if sc.get(col) is not None else "NULL")

    cols += ["priority_level", "last_visit_date", "visit_count", "is_deleted", "created_at", "notes"]
    vals += [esc(sc.get("priority_level", "Medically Fit")), "NULL", "0", "false", esc(now), esc(body.notes)]

    col_str = ", ".join(cols)
    val_str = ", ".join(vals)
    return f"INSERT INTO workspace.suraksha.patients ({col_str}) VALUES ({val_str})"


def _build_update(pid: str, body: PatientIn, sc: dict, now: str) -> str:
    data  = body.model_dump()
    parts = []

    for col, typ in _EDITABLE_COLS:
        parts.append(f"{col} = {_sql_val(data.get(col), typ)}")

    for col in ("risk_complication", "risk_home_delivery",
                "risk_immunization", "risk_child_mortality"):
        v = sc.get(col)
        parts.append(f"{col} = {'NULL' if v is None else v}")

    parts += [
        f"priority_level  = {esc(sc.get('priority_level', 'Medically Fit'))}",
        f"last_visit_date = {esc(now)}",
        f"visit_count     = visit_count + 1",
        f"notes           = {esc(body.notes)}",
    ]

    set_clause = ",\n            ".join(parts)
    return f"""
        UPDATE workspace.suraksha.patients
        SET {set_clause}
        WHERE patient_id = {esc(pid)}
    """


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
async def list_patients(psu: Optional[str] = None, priority_level: Optional[str] = None):
    where = "WHERE is_deleted = false"
    if psu:
        where += f" AND PSU_ID = {esc(psu)}"
    if priority_level:
        where += f" AND priority_level = {esc(priority_level)}"
    return run_query(f"""
        SELECT * FROM workspace.suraksha.patients
        {where}
        ORDER BY overall_risk DESC NULLS LAST
    """)


@router.post("")
async def create_patient(body: PatientIn):
    pid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    sc  = score_patient(body.model_dump())
    execute(_build_insert(pid, body, sc, now))
    return {"patient_id": pid, **sc}


@router.get("/{patient_id}")
async def get_patient(patient_id: str):
    rows = run_query(f"""
        SELECT * FROM workspace.suraksha.patients
        WHERE patient_id = {esc(patient_id)} AND is_deleted = false
    """)
    if not rows:
        raise HTTPException(status_code=404, detail="Patient not found")
    return rows[0]


@router.put("/{patient_id}")
async def update_patient(patient_id: str, body: PatientIn):
    if not run_query(f"""
        SELECT 1 FROM workspace.suraksha.patients
        WHERE patient_id = {esc(patient_id)} AND is_deleted = false
    """):
        raise HTTPException(status_code=404, detail="Patient not found")

    now = datetime.utcnow().isoformat()
    sc  = score_patient(body.model_dump())

    comp = sc.get('risk_complication')
    execute(f"""
        INSERT INTO workspace.suraksha.visit_log VALUES (
            {esc(str(uuid.uuid4()))},
            {esc(patient_id)},
            {esc(now)},
            {esc(body.notes or '')},
            {'NULL' if comp is None else comp}
        )
    """)
    execute(_build_update(patient_id, body, sc, now))
    return {"patient_id": patient_id, **sc}


@router.delete("/{patient_id}")
async def delete_patient(patient_id: str):
    if not run_query(f"""
        SELECT 1 FROM workspace.suraksha.patients
        WHERE patient_id = {esc(patient_id)} AND is_deleted = false
    """):
        raise HTTPException(status_code=404, detail="Patient not found")
    execute(f"""
        UPDATE workspace.suraksha.patients
        SET is_deleted = true
        WHERE patient_id = {esc(patient_id)}
    """)
    return {"deleted": True}


@router.get("/{patient_id}/visits")
async def get_visits(patient_id: str):
    return run_query(f"""
        SELECT * FROM workspace.suraksha.visit_log
        WHERE patient_id = {esc(patient_id)}
        ORDER BY visit_date DESC
    """)
