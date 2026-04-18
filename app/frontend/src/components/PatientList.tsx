import { useState } from 'react'
import { Pencil, Trash2, UserPlus } from 'lucide-react'
import { Patient } from '../types'
import { priorityColor, priorityBg, fmt, fmtDate, sortByPriority } from '../utils'

interface Props {
  patients: Patient[]
  onSelect: (p: Patient) => void
  onEdit:   (p: Patient) => void
  onDelete: (id: string) => void
  onAdd:    () => void
}

const PRIORITY_LEVELS = ['High Priority', 'Routine Care', 'Medically Fit'] as const

export default function PatientList({ patients, onSelect, onEdit, onDelete, onAdd }: Props) {
  const [filterLevel, setFilterLevel] = useState('')

  const sorted   = sortByPriority(patients)
  const filtered = filterLevel ? sorted.filter(p => p.priority_level === filterLevel) : sorted

  return (
    <div className="patients-view">
      <div className="list-toolbar">
        <h2 className="section-title">Patients — Ranked by Priority</h2>
        <div className="toolbar-right">
          <select className="filter-select" value={filterLevel} onChange={e => setFilterLevel(e.target.value)}>
            <option value="">All Priority Levels</option>
            {PRIORITY_LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
          <button className="btn-primary" onClick={onAdd} style={{ display:'flex', alignItems:'center', gap:6 }}>
            <UserPlus size={15} /> Add Patient
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="card empty-card">
          {patients.length === 0
            ? 'No patients registered yet. Click "+ Add Patient" to begin.'
            : 'No patients match the selected filter.'}
        </div>
      ) : (
        <div className="card">
          <table className="data-table patient-table">
            <thead>
              <tr>
                <th>#</th><th>Name</th><th>PSU</th><th>Age</th><th>Weeks</th>
                <th>Priority</th>
                <th title="Complication Risk — Priority 1">Comp. ①</th>
                <th title="Home Delivery Risk — Priority 2">Home Del. ②</th>
                <th title="Immunization Risk — Priority 3">Immun. ③</th>
                <th>Last Visit</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => (
                <tr key={p.patient_id} className="patient-row" onClick={() => onSelect(p)}>
                  <td style={{ color:'#9ca3af', fontWeight:600 }}>{i + 1}</td>
                  <td><strong>{p.name || `#${String(p.patient_id).slice(0,6)}`}</strong></td>
                  <td>{p.PSU_ID}</td>
                  <td>{p.age}</td>
                  <td>{p.weeks_pregnant ?? '—'}</td>
                  <td>
                    <span className="badge" style={{
                      background: priorityBg(p.priority_level || ''),
                      color: priorityColor(p.priority_level || ''),
                    }}>
                      {p.priority_level || '—'}
                    </span>
                  </td>
                  <td>
                    <div style={{ display:'flex', alignItems:'center', gap:4 }}>
                      <div style={{ width:36, height:5, background:'#fee2e2', borderRadius:3, overflow:'hidden' }}>
                        <div style={{ height:'100%', width:`${Math.round((p.risk_complication??0)*100)}%`, background:'#ef4444', borderRadius:3 }} />
                      </div>
                      <span style={{ fontSize:12, color:'#ef4444', fontWeight:600 }}>{fmt(p.risk_complication)}</span>
                    </div>
                  </td>
                  <td>
                    <div style={{ display:'flex', alignItems:'center', gap:4 }}>
                      <div style={{ width:36, height:5, background:'#ffedd5', borderRadius:3, overflow:'hidden' }}>
                        <div style={{ height:'100%', width:`${Math.round((p.risk_home_delivery??0)*100)}%`, background:'#f97316', borderRadius:3 }} />
                      </div>
                      <span style={{ fontSize:12, color:'#f97316', fontWeight:600 }}>{fmt(p.risk_home_delivery)}</span>
                    </div>
                  </td>
                  <td>
                    <div style={{ display:'flex', alignItems:'center', gap:4 }}>
                      <div style={{ width:36, height:5, background:'#f3e8ff', borderRadius:3, overflow:'hidden' }}>
                        <div style={{ height:'100%', width:`${Math.round((p.risk_immunization??0)*100)}%`, background:'#a855f7', borderRadius:3 }} />
                      </div>
                      <span style={{ fontSize:12, color:'#a855f7', fontWeight:600 }}>{fmt(p.risk_immunization)}</span>
                    </div>
                  </td>
                  <td>{fmtDate(p.last_visit_date)}</td>
                  <td onClick={e => e.stopPropagation()}>
                    <button className="icon-btn" title="Edit"   onClick={() => onEdit(p)}><Pencil size={14} /></button>
                    <button className="icon-btn" title="Delete" onClick={() => onDelete(p.patient_id)}><Trash2 size={14} color="#dc2626" /></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
