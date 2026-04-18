export interface Patient {
  patient_id: string
  name: string
  PSU_ID: string
  age: number
  weeks_pregnant: number | null
  rural: string
  social_group_code: string
  marital_status: string
  highest_qualification: string
  cooking_fuel: string
  toilet_used: string
  is_television: string
  is_telephone: string
  house_structure: string
  drinking_water_source: string
  source_of_anc: string
  w_preg_no: number | null
  no_of_anc: number | null
  had_anc_registration: number
  // ANC symptoms
  swelling_of_hand_feet_face: string
  hypertension_high_bp: string
  excessive_bleeding: string
  paleness_giddiness_weakness: string
  visual_disturbance: string
  excessive_vomiting: string
  convulsion_not_from_fever: string
  // 6 delivery complication flags
  premature_labour: string
  prolonged_labour: string
  obstructed_labour: string
  excessive_bleeding_during_birth: string
  convulsion_high_bp: string
  breech_presentation: string
  // Risk scores — priority order: 1 → 2 → 3
  risk_complication: number | null    // Priority 1
  risk_home_delivery: number | null   // Priority 2
  risk_immunization: number | null    // Priority 3
  risk_child_mortality: number | null
  priority_level: string              // "High Priority" | "Routine Care" | "Medically Fit"
  last_visit_date: string | null
  visit_count: number
  created_at: string
  notes: string | null
}

export interface PSURow {
  PSU_ID: string
  total: number
  high: number
  medium: number
  low: number
  avg_risk: number
}

export interface Visit {
  visit_id: string
  visit_date: string
  notes: string
  overall_risk: number | null
}

export interface FormState {
  name: string
  PSU_ID: string
  age: string
  weeks_pregnant: string
  rural: string
  social_group_code: string
  marital_status: string
  highest_qualification: string
  cooking_fuel: string
  toilet_used: string
  is_television: string
  is_telephone: string
  house_structure: string
  drinking_water_source: string
  source_of_anc: string
  w_preg_no: string
  no_of_anc: string
  had_anc_registration: string
  swelling_of_hand_feet_face: string
  hypertension_high_bp: string
  excessive_bleeding: string
  paleness_giddiness_weakness: string
  visual_disturbance: string
  excessive_vomiting: string
  convulsion_not_from_fever: string
  notes: string
}

export const EMPTY_FORM: FormState = {
  name: '', PSU_ID: '', age: '', weeks_pregnant: '',
  rural: 'Rural', social_group_code: 'OBC', marital_status: 'Married',
  highest_qualification: 'Primary', cooking_fuel: 'Wood/Dung Cake',
  toilet_used: 'Open Defecation', is_television: 'No', is_telephone: 'No',
  house_structure: 'Kutcha', drinking_water_source: 'Hand Pump',
  source_of_anc: 'Government', w_preg_no: '', no_of_anc: '',
  had_anc_registration: '0',
  swelling_of_hand_feet_face: 'No', hypertension_high_bp: 'No',
  excessive_bleeding: 'No', paleness_giddiness_weakness: 'No',
  visual_disturbance: 'No', excessive_vomiting: 'No',
  convulsion_not_from_fever: 'No', notes: '',
}
