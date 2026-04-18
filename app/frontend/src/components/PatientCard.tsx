import { ArrowRight } from 'lucide-react'
import { Patient } from '../types'
import { priorityColor, priorityBg, fmt } from '../utils'

interface Props {
  patient: Patient
  rank:    number
  onClick: (p: Patient) => void
}

const PRIORITY_BORDER: Record<string, string> = {
  'High Priority': '#fca5a5',
  'Routine Care':  '#fcd34d',
  'Medically Fit': '#86efac',
}

const PRIORITY_GRADIENT: Record<string, string> = {
  'High Priority': 'linear-gradient(135deg,#fff1f2 0%,#ffe4e6 100%)',
  'Routine Care':  'linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%)',
  'Medically Fit': 'linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%)',
}

// 6 delivery complication flags
const COMPLICATION_LABELS: { key: keyof Patient; label: string }[] = [
  { key: 'premature_labour',                label: 'Premature Labour' },
  { key: 'prolonged_labour',                label: 'Prolonged Labour' },
  { key: 'obstructed_labour',               label: 'Obstructed Labour' },
  { key: 'excessive_bleeding_during_birth', label: 'Excessive Bleeding' },
  { key: 'convulsion_high_bp',              label: 'Eclampsia' },
  { key: 'breech_presentation',             label: 'Breech' },
]

function MiniBar({ value, color }: { value: number | null; color: string }) {
  return (
    <div style={{ flex:1, height:5, background:'#e2e8f0', borderRadius:3, overflow:'hidden' }}>
      <div style={{ height:'100%', width:`${Math.round((value ?? 0)*100)}%`, background:color, borderRadius:3, transition:'width 0.6s ease' }} />
    </div>
  )
}

function Flag({ label }: { label: string }) {
  return (
    <span style={{ fontSize:11, padding:'2px 8px', borderRadius:20, background:'#fee2e2', color:'#b91c1c', fontWeight:600 }}>
      {label}
    </span>
  )
}

export default function PatientCard({ patient: p, rank, onClick }: Props) {
  const level  = p.priority_level || 'Medically Fit'
  const color  = priorityColor(level)
  const border = PRIORITY_BORDER[level] || '#e2e8f0'
  const grad   = PRIORITY_GRADIENT[level] || '#fff'

  // ANC warning flags
  const flags: string[] = []
  if (p.hypertension_high_bp === 'Yes')        flags.push('High BP')
  if (p.swelling_of_hand_feet_face === 'Yes')  flags.push('Swelling')
  if (p.excessive_bleeding === 'Yes')          flags.push('Bleeding')
  if (p.paleness_giddiness_weakness === 'Yes') flags.push('Anaemia')
  if (p.convulsion_not_from_fever === 'Yes')   flags.push('Convulsions')
  if ((Number(p.no_of_anc) || 0) === 0)       flags.push('No ANC')

  // Active delivery complication flags
  const compFlags = COMPLICATION_LABELS
    .filter(c => p[c.key] === 'Yes')
    .map(c => c.label)

  const allFlags = [...compFlags, ...flags]

  return (
    <div className="patient-card" style={{ background:grad, borderColor:border }} onClick={() => onClick(p)}>

      {/* Header: rank badge + priority + PSU */}
      <div className="pc-header">
        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
          <span style={{ fontSize:11, fontWeight:700, color:'#9ca3af', minWidth:22 }}>#{rank}</span>
          <span className="pc-risk-badge" style={{ background:color }}>{level}</span>
        </div>
        <span className="pc-psu">PSU {p.PSU_ID}</span>
      </div>

      {/* Identity */}
      <div className="pc-identity">
        <span className="pc-name">{p.name || `Patient #${String(p.patient_id).slice(0,6)}`}</span>
        <span className="pc-meta">
          Age {p.age}
          {p.w_preg_no ? ` · Preg #${p.w_preg_no}` : ''}
          {p.rural ? ` · ${p.rural}` : ''}
        </span>
      </div>

      {/* Priority-ranked risk bars */}
      <div className="pc-bars">
        <div className="pc-bar-row">
          <span style={{ fontWeight:600, color:'#374151' }}>① Complication</span>
          <MiniBar value={p.risk_complication} color="#ef4444" />
          <span style={{ color:'#ef4444', fontWeight:700 }}>{fmt(p.risk_complication)}</span>
        </div>
        <div className="pc-bar-row">
          <span>② Home Delivery</span>
          <MiniBar value={p.risk_home_delivery} color="#f97316" />
          <span style={{ color:'#f97316', fontWeight:600 }}>{fmt(p.risk_home_delivery)}</span>
        </div>
        <div className="pc-bar-row">
          <span>③ Immunization</span>
          <MiniBar value={p.risk_immunization} color="#a855f7" />
          <span style={{ color:'#a855f7', fontWeight:600 }}>{fmt(p.risk_immunization)}</span>
        </div>
      </div>

      {/* Complication + ANC flags */}
      {allFlags.length > 0 && (
        <div className="pc-flags">
          {allFlags.slice(0, 4).map(f => <Flag key={f} label={f} />)}
          {allFlags.length > 4 && <span style={{ fontSize:11, color:'#6b7280' }}>+{allFlags.length - 4} more</span>}
        </div>
      )}

      {/* Footer */}
      <div className="pc-footer">
        <span style={{ fontSize:11, color:'#6b7280' }}>
          {p.risk_child_mortality != null ? `Child mortality: ${fmt(p.risk_child_mortality)}` : ''}
        </span>
        <span className="pc-cta" style={{ display:'flex', alignItems:'center', gap:4 }}>
          View Details <ArrowRight size={13} />
        </span>
      </div>
    </div>
  )
}
