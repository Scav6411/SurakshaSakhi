import { Patient, PSURow, Visit, FormState } from '../types'

const toPayload = (f: FormState) => ({
  name:                        f.name,
  PSU_ID:                      f.PSU_ID,
  age:                         parseInt(f.age) || 0,
  weeks_pregnant:              parseInt(f.weeks_pregnant) || null,
  rural:                       f.rural,
  social_group_code:           f.social_group_code,
  marital_status:              f.marital_status,
  highest_qualification:       f.highest_qualification,
  cooking_fuel:                f.cooking_fuel,
  toilet_used:                 f.toilet_used,
  is_television:               f.is_television,
  is_telephone:                f.is_telephone,
  house_structure:             f.house_structure,
  drinking_water_source:       f.drinking_water_source,
  source_of_anc:               f.source_of_anc,
  w_preg_no:                   parseInt(f.w_preg_no) || null,
  no_of_anc:                   parseInt(f.no_of_anc) || null,
  had_anc_registration:        parseInt(f.had_anc_registration) || 0,
  swelling_of_hand_feet_face:  f.swelling_of_hand_feet_face,
  hypertension_high_bp:        f.hypertension_high_bp,
  excessive_bleeding:          f.excessive_bleeding,
  paleness_giddiness_weakness: f.paleness_giddiness_weakness,
  visual_disturbance:          f.visual_disturbance,
  excessive_vomiting:          f.excessive_vomiting,
  convulsion_not_from_fever:   f.convulsion_not_from_fever,
  notes:                       f.notes,
})

const json = (r: Response) => r.json()

export const fetchPatients = (psu?: string, priorityLevel?: string): Promise<Patient[]> => {
  const params = new URLSearchParams()
  if (psu)           params.set('psu', psu)
  if (priorityLevel) params.set('priority_level', priorityLevel)
  const qs = params.toString()
  return fetch(`/api/patients${qs ? '?' + qs : ''}`).then(json)
}

export const fetchPatient = (id: string): Promise<Patient> =>
  fetch(`/api/patients/${id}`).then(json)

export const createPatient = (form: FormState): Promise<{ patient_id: string }> =>
  fetch('/api/patients', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toPayload(form)),
  }).then(json)

export const updatePatient = (id: string, form: FormState): Promise<{ patient_id: string }> =>
  fetch(`/api/patients/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toPayload(form)),
  }).then(json)

export const deletePatient = (id: string): Promise<{ deleted: boolean }> =>
  fetch(`/api/patients/${id}`, { method: 'DELETE' }).then(json)

export const fetchVisits = (id: string): Promise<Visit[]> =>
  fetch(`/api/patients/${id}/visits`).then(json)

export const fetchPSUSummary = (): Promise<PSURow[]> =>
  fetch('/api/dashboard/psu').then(json)

export interface GenieResponse {
  text:            string
  columns:         string[] | null
  rows:            string[][] | null
  sql:             string | null
  conversation_id: string
}

export const genieQuery = (query: string, conversationId?: string): Promise<GenieResponse> =>
  fetch('/api/genie/query', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query, conversation_id: conversationId ?? null }),
  }).then(json)
