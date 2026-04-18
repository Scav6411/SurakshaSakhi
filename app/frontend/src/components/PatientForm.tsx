import { useState } from 'react'
import { FormState } from '../types'

interface Props {
  initial: FormState
  title: string
  onSave: (form: FormState) => Promise<void>
  onClose: () => void
}

function SF({ label, name, value, onChange, options }: {
  label: string; name: string; value: string; options: string[]
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void
}) {
  return (
    <div className="form-field">
      <label>{label}</label>
      <select name={name} value={value} onChange={onChange}>
        {options.map(o => <option key={o}>{o}</option>)}
      </select>
    </div>
  )
}

function NF({ label, name, value, onChange, placeholder }: {
  label: string; name: string; value: string; placeholder?: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
}) {
  return (
    <div className="form-field">
      <label>{label}</label>
      <input type="text" name={name} value={value} onChange={onChange} placeholder={placeholder} />
    </div>
  )
}

const YN  = ['Yes', 'No']
const SYM = ['Yes', 'No', 'Not_Reported']

export default function PatientForm({ initial, title, onSave, onClose }: Props) {
  const [form, setForm]   = useState<FormState>(initial)
  const [saving, setSaving] = useState(false)

  const ch = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) =>
    setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    try { await onSave(form) } finally { setSaving(false) }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{title}</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        <form onSubmit={submit} className="patient-form">
          <div className="form-section">
            <h3>Basic Info</h3>
            <div className="form-grid">
              <NF label="Name"           name="name"           value={form.name}           onChange={ch} placeholder="Patient name" />
              <NF label="PSU ID (Village)" name="PSU_ID"       value={form.PSU_ID}         onChange={ch} placeholder="e.g. 101" />
              <NF label="Age"            name="age"            value={form.age}            onChange={ch} placeholder="Years" />
              <NF label="Weeks Pregnant" name="weeks_pregnant" value={form.weeks_pregnant} onChange={ch} placeholder="Weeks" />
            </div>
          </div>

          <div className="form-section">
            <h3>Demographics</h3>
            <div className="form-grid">
              <SF label="Rural / Urban"  name="rural"               value={form.rural}               onChange={ch} options={['Rural', 'Urban']} />
              <SF label="Social Group"   name="social_group_code"   value={form.social_group_code}   onChange={ch} options={['SC', 'ST', 'OBC', 'General']} />
              <SF label="Marital Status" name="marital_status"      value={form.marital_status}      onChange={ch} options={['Married', 'Widowed', 'Divorced', 'Never Married']} />
              <SF label="Education"      name="highest_qualification" value={form.highest_qualification} onChange={ch} options={['Illiterate', 'Primary', 'Middle', 'Secondary', 'Higher Secondary', 'Graduate+']} />
            </div>
          </div>

          <div className="form-section">
            <h3>Household</h3>
            <div className="form-grid">
              <SF label="Cooking Fuel"  name="cooking_fuel"          value={form.cooking_fuel}          onChange={ch} options={['Wood/Dung Cake', 'Kerosene', 'LPG/PNG', 'Other']} />
              <SF label="Toilet"        name="toilet_used"           value={form.toilet_used}           onChange={ch} options={['Open Defecation', 'Shared Toilet', 'Own Toilet']} />
              <SF label="Television"    name="is_television"         value={form.is_television}         onChange={ch} options={YN} />
              <SF label="Telephone"     name="is_telephone"          value={form.is_telephone}          onChange={ch} options={YN} />
              <SF label="House Type"    name="house_structure"       value={form.house_structure}       onChange={ch} options={['Kutcha', 'Semi-Pucca', 'Pucca']} />
              <SF label="Water Source"  name="drinking_water_source" value={form.drinking_water_source} onChange={ch} options={['Tap', 'Well', 'Hand Pump', 'Other']} />
            </div>
          </div>

          <div className="form-section">
            <h3>Pregnancy & ANC</h3>
            <div className="form-grid">
              <NF label="Pregnancy No."  name="w_preg_no"           value={form.w_preg_no}           onChange={ch} placeholder="1st, 2nd…" />
              <NF label="ANC Visits"     name="no_of_anc"           value={form.no_of_anc}           onChange={ch} placeholder="Count" />
              <SF label="ANC Registered" name="had_anc_registration" value={form.had_anc_registration} onChange={ch} options={['1', '0']} />
              <SF label="ANC Source"     name="source_of_anc"       value={form.source_of_anc}       onChange={ch} options={['Government', 'Private', 'ASHA', 'None']} />
            </div>
          </div>

          <div className="form-section">
            <h3>ANC Symptoms</h3>
            <div className="form-grid">
              <SF label="Swelling (hands/feet/face)" name="swelling_of_hand_feet_face"  value={form.swelling_of_hand_feet_face}  onChange={ch} options={SYM} />
              <SF label="Hypertension / High BP"     name="hypertension_high_bp"         value={form.hypertension_high_bp}         onChange={ch} options={SYM} />
              <SF label="Excessive Bleeding"         name="excessive_bleeding"           value={form.excessive_bleeding}           onChange={ch} options={SYM} />
              <SF label="Paleness / Giddiness"       name="paleness_giddiness_weakness"  value={form.paleness_giddiness_weakness}  onChange={ch} options={SYM} />
              <SF label="Visual Disturbance"         name="visual_disturbance"           value={form.visual_disturbance}           onChange={ch} options={SYM} />
              <SF label="Excessive Vomiting"         name="excessive_vomiting"           value={form.excessive_vomiting}           onChange={ch} options={SYM} />
              <SF label="Convulsion (non-fever)"     name="convulsion_not_from_fever"    value={form.convulsion_not_from_fever}    onChange={ch} options={SYM} />
            </div>
          </div>

          <div className="form-section">
            <h3>Notes</h3>
            <div className="form-grid">
              <NF label="Visit Notes" name="notes" value={form.notes} onChange={ch} placeholder="Observations from this visit…" />
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn-primary" disabled={saving}>
              {saving ? 'Saving…' : 'Save Patient'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
