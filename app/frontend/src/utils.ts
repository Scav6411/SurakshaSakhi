import { Patient, FormState } from './types'

// Priority level colours — "High Priority" | "Routine Care" | "Medically Fit"
export const priorityColor = (level: string): string =>
  level === 'High Priority' ? '#dc2626'
  : level === 'Routine Care' ? '#d97706'
  : '#059669'

export const priorityBg = (level: string): string =>
  level === 'High Priority' ? '#fef2f2'
  : level === 'Routine Care' ? '#fffbeb'
  : '#f0fdf4'

export const priorityLabel = (level: string): string =>
  level === 'High Priority' ? 'High Priority'
  : level === 'Routine Care' ? 'Routine Care'
  : 'Medically Fit'

// Sort patients: complication desc → home delivery desc → immunization desc
export const sortByPriority = (patients: Patient[]): Patient[] =>
  [...patients].sort((a, b) => {
    const c = (b.risk_complication ?? 0) - (a.risk_complication ?? 0)
    if (c !== 0) return c
    const h = (b.risk_home_delivery ?? 0) - (a.risk_home_delivery ?? 0)
    if (h !== 0) return h
    return (b.risk_immunization ?? 0) - (a.risk_immunization ?? 0)
  })

export const fmt = (v: number | null): string =>
  v == null ? '—' : (v * 100).toFixed(0) + '%'

export const fmtDate = (s: string | null): string =>
  s ? new Date(s).toLocaleDateString('en-IN') : '—'

export const patientToForm = (p: Patient): FormState => ({
  name: p.name || '',
  PSU_ID: p.PSU_ID || '',
  age: String(p.age || ''),
  weeks_pregnant: String(p.weeks_pregnant || ''),
  rural: p.rural || 'Rural',
  social_group_code: p.social_group_code || 'OBC',
  marital_status: p.marital_status || 'Married',
  highest_qualification: p.highest_qualification || 'Primary',
  cooking_fuel: p.cooking_fuel || 'Wood/Dung Cake',
  toilet_used: p.toilet_used || 'Open Defecation',
  is_television: p.is_television || 'No',
  is_telephone: p.is_telephone || 'No',
  house_structure: p.house_structure || 'Kutcha',
  drinking_water_source: p.drinking_water_source || 'Hand Pump',
  source_of_anc: p.source_of_anc || 'Government',
  w_preg_no: String(p.w_preg_no || ''),
  no_of_anc: String(p.no_of_anc || ''),
  had_anc_registration: String(p.had_anc_registration || '0'),
  swelling_of_hand_feet_face: p.swelling_of_hand_feet_face || 'No',
  hypertension_high_bp: p.hypertension_high_bp || 'No',
  excessive_bleeding: p.excessive_bleeding || 'No',
  paleness_giddiness_weakness: p.paleness_giddiness_weakness || 'No',
  visual_disturbance: p.visual_disturbance || 'No',
  excessive_vomiting: p.excessive_vomiting || 'No',
  convulsion_not_from_fever: p.convulsion_not_from_fever || 'No',
  notes: '',
})
