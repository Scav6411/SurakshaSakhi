import { ChevronLeft, Pencil, Trash2, Calendar, Activity, AlertTriangle } from 'lucide-react'
import { Patient, Visit } from '../types'
import { priorityColor, priorityBg, fmt, fmtDate } from '../utils'
import RiskBar from './RiskBar'

interface Props {
  patient: Patient
  visits:  Visit[]
  onBack:   () => void
  onEdit:   (p: Patient) => void
  onDelete: (id: string) => void
}

// 6 delivery complications
const COMPLICATIONS: { key: keyof Patient; label: string }[] = [
  { key: 'premature_labour',                label: 'Premature Labour' },
  { key: 'prolonged_labour',                label: 'Prolonged Labour' },
  { key: 'obstructed_labour',               label: 'Obstructed Labour' },
  { key: 'excessive_bleeding_during_birth', label: 'Excessive Bleeding at Birth' },
  { key: 'convulsion_high_bp',              label: 'Eclampsia / High BP Convulsion' },
  { key: 'breech_presentation',             label: 'Breech Presentation' },
]

function ComplicationRow({ label, value }: { label: string; value: string }) {
  const isYes = value === 'Yes'
  return (
    <div style={{
      display:'flex', justifyContent:'space-between', alignItems:'center',
      padding:'5px 0', borderBottom:'1px solid #f1f5f9',
    }}>
      <span style={{ fontSize:13, color: isYes ? '#b91c1c' : '#374151' }}>{label}</span>
      <span style={{
        fontSize:11, fontWeight:700, padding:'2px 10px', borderRadius:20,
        background: isYes ? '#fee2e2' : '#f0fdf4',
        color: isYes ? '#dc2626' : '#059669',
      }}>
        {value === 'Yes' ? 'YES' : value === 'No' ? 'No' : '—'}
      </span>
    </div>
  )
}

export default function PatientDetail({ patient, visits, onBack, onEdit, onDelete }: Props) {
  const level = patient.priority_level || 'Medically Fit'

  return (
    <div className="detail-view">
      <button className="back-btn" onClick={onBack} style={{ display:'flex', alignItems:'center', gap:4 }}>
        <ChevronLeft size={16} /> Back to List
      </button>

      <div className="detail-header">
        <div>
          <h2 className="detail-name">{patient.name}</h2>
          <p className="detail-meta">
            PSU {patient.PSU_ID}&nbsp;·&nbsp;Age {patient.age}
            {patient.weeks_pregnant ? ` · ${patient.weeks_pregnant} weeks` : ''}
            &nbsp;·&nbsp;{patient.rural}&nbsp;·&nbsp;Visits: {patient.visit_count || 0}
          </p>
        </div>
        <div className="detail-actions">
          <button className="btn-secondary" onClick={() => onEdit(patient)} style={{ display:'flex', alignItems:'center', gap:6 }}>
            <Pencil size={14} /> Edit / Record Visit
          </button>
          <button className="btn-danger" onClick={() => onDelete(patient.patient_id)} style={{ display:'flex', alignItems:'center', gap:6 }}>
            <Trash2 size={14} /> Delete
          </button>
        </div>
      </div>

      <div className="detail-grid">

        {/* Risk scores — priority ranked */}
        <div className="card risk-card">
          <h3 className="card-title" style={{ display:'flex', alignItems:'center', gap:6 }}>
            <Activity size={15} /> Risk Scores
          </h3>
          <div className="risk-level-badge" style={{ background: priorityBg(level), color: priorityColor(level) }}>
            {level}
          </div>
          <RiskBar label="① Complication Risk"  value={patient.risk_complication}   color="#ef4444" />
          <RiskBar label="② Home Delivery Risk"  value={patient.risk_home_delivery}  color="#f97316" />
          <RiskBar label="③ Immunization Risk"   value={patient.risk_immunization}   color="#a855f7" />
          {patient.risk_child_mortality != null && (
            <RiskBar label="Child Mortality Risk" value={patient.risk_child_mortality} color="#64748b" />
          )}
        </div>

        {/* Delivery complications — the 6 labels */}
        <div className="card">
          <h3 className="card-title" style={{ display:'flex', alignItems:'center', gap:6 }}>
            <AlertTriangle size={15} color="#ef4444" /> Delivery Complications
          </h3>
          {COMPLICATIONS.map(c => (
            <ComplicationRow
              key={c.key}
              label={c.label}
              value={String(patient[c.key] || '—')}
            />
          ))}
        </div>

        {/* Patient details */}
        <div className="card">
          <h3 className="card-title">Patient Details</h3>
          <dl className="detail-dl">
            <dt>Social Group</dt><dd>{patient.social_group_code}</dd>
            <dt>Education</dt>   <dd>{patient.highest_qualification}</dd>
            <dt>Toilet</dt>      <dd>{patient.toilet_used}</dd>
            <dt>Cooking Fuel</dt><dd>{patient.cooking_fuel}</dd>
            <dt>ANC Visits</dt>  <dd>{patient.no_of_anc ?? '—'}</dd>
            <dt>Pregnancy No.</dt><dd>{patient.w_preg_no ?? '—'}</dd>
            <dt>Last Visit</dt>  <dd>{fmtDate(patient.last_visit_date)}</dd>
            <dt>Registered</dt>  <dd>{fmtDate(patient.created_at)}</dd>
          </dl>
        </div>
      </div>

      {visits.length > 0 && (
        <div className="card">
          <h3 className="card-title" style={{ display:'flex', alignItems:'center', gap:6 }}>
            <Calendar size={15} /> Visit History
          </h3>
          <table className="data-table">
            <thead>
              <tr><th>Date</th><th>Complication Risk at Visit</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {visits.map(v => (
                <tr key={v.visit_id}>
                  <td>{fmtDate(v.visit_date)}</td>
                  <td>{v.overall_risk != null ? `${(Number(v.overall_risk) * 100).toFixed(0)}%` : '—'}</td>
                  <td>{v.notes || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
