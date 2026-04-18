from typing import Optional
from pydantic import BaseModel


class PatientIn(BaseModel):
    # ── Required ──────────────────────────────────────────────────────────
    name:    str
    PSU_ID:  str
    age:     int

    # ── Demographics ──────────────────────────────────────────────────────
    weeks_pregnant:             Optional[int]   = None
    rural:                      Optional[str]   = None
    marital_status:             Optional[str]   = None
    social_group_code:          Optional[str]   = None
    highest_qualification:      Optional[str]   = None
    w_preg_no:                  Optional[int]   = None
    mother_age_when_baby_was_born: Optional[float] = None
    order_of_birth:             Optional[float] = None

    # ── ANC / pregnancy symptoms ──────────────────────────────────────────
    no_of_anc:                      Optional[int]   = None
    source_of_anc:                  Optional[str]   = None
    had_anc_registration:           Optional[int]   = None
    no_of_tt_injections:            Optional[float] = None
    consumption_of_ifa:             Optional[str]   = None
    swelling_of_hand_feet_face:     Optional[str]   = None
    hypertension_high_bp:           Optional[str]   = None
    excessive_bleeding:             Optional[str]   = None
    paleness_giddiness_weakness:    Optional[str]   = None
    visual_disturbance:             Optional[str]   = None
    excessive_vomiting:             Optional[str]   = None
    convulsion_not_from_fever:      Optional[str]   = None

    # ── Household ─────────────────────────────────────────────────────────
    cooking_fuel:           Optional[str] = None
    toilet_used:            Optional[str] = None
    is_telephone:           Optional[str] = None
    is_television:          Optional[str] = None
    house_structure:        Optional[str] = None
    drinking_water_source:  Optional[str] = None

    # ── Delivery / birth (Model 2, 3, 4 features) ────────────────────────
    where_del_took_place:           Optional[str] = None
    type_of_delivery:               Optional[str] = None
    type_of_birth:                  Optional[str] = None
    gender:                         Optional[str] = None
    who_conducted_del_at_home:      Optional[str] = None
    check_up_with_48_hours_of_del:  Optional[str] = None
    first_breast_feeding:           Optional[str] = None
    weight_of_baby_kg:              Optional[float] = None
    weight_of_baby_grams:           Optional[float] = None

    # ── Model 1 targets (delivery complications) ─────────────────────────
    premature_labour:               Optional[str] = None
    prolonged_labour:               Optional[str] = None
    obstructed_labour:              Optional[str] = None
    excessive_bleeding_during_birth: Optional[str] = None
    convulsion_high_bp:             Optional[str] = None
    breech_presentation:            Optional[str] = None

    # ── Model 3 immunization label columns ────────────────────────────────
    bcg_vaccine:                    Optional[str]   = None
    no_of_polio_doses_ri:           Optional[float] = None
    no_of_dpt_injection:            Optional[float] = None
    measles:                        Optional[str]   = None
    ever_vacination_taken_bye_baby: Optional[str]   = None

    # ── Model 4 target ────────────────────────────────────────────────────
    kind_of_birth:                  Optional[str] = None

    # ── App only ──────────────────────────────────────────────────────────
    notes: Optional[str] = None
