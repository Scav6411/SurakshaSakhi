import { useState } from 'react'
import React from 'react'
import { AlertTriangle, AlertCircle, Eye, Users, MapPin, CheckCircle } from 'lucide-react'
import { Patient, PSURow } from '../types'
import { priorityColor, sortByPriority } from '../utils'
import PatientCard from './PatientCard'
import DatabricksDashboardEmbed from './DatabricksDashboardEmbed'

interface Props {
  patients:        Patient[]
  psuData:         PSURow[]
  onSelectPatient: (p: Patient) => void
}

function StatCard({ label, value, sub, color, icon }: {
  label: string; value: number; sub?: string; color: string; icon?: React.ReactNode
}) {
  return (
    <div className="stat-card-new" style={{ borderTopColor: color }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom: 4 }}>
        <div className="stat-card-value" style={{ color }}>{value}</div>
        {icon && <span style={{ color, opacity: 0.6 }}>{icon}</span>}
      </div>
      <div className="stat-card-label">{label}</div>
      {sub && <div className="stat-card-sub">{sub}</div>}
    </div>
  )
}

type Filter = 'High Priority' | 'Routine Care' | 'All'

export default function Dashboard({ patients, psuData, onSelectPatient }: Props) {
  const [cardFilter, setCardFilter] = useState<Filter>('High Priority')

  const highPriority = patients.filter(p => p.priority_level === 'High Priority')
  const routineCare  = patients.filter(p => p.priority_level === 'Routine Care')
  const medicallyFit = patients.filter(p => p.priority_level === 'Medically Fit')

  const displayCards = sortByPriority(
    cardFilter === 'All' ? patients : patients.filter(p => p.priority_level === cardFilter)
  ).slice(0, 18)

  return (
    <div className="dashboard-new">

      <div className="stat-row">
        <StatCard label="Total Patients"  value={patients.length}     color="#6366f1" sub="registered"     icon={<Users size={18} />} />
        <StatCard label="High Priority"   value={highPriority.length} color="#ef4444"
          sub={`${Math.round(highPriority.length / (patients.length || 1) * 100)}% of total`}
          icon={<AlertTriangle size={18} />} />
        <StatCard label="Routine Care"    value={routineCare.length}  color="#f97316" sub="needs monitoring" icon={<AlertCircle size={18} />} />
        <StatCard label="Medically Fit"   value={medicallyFit.length} color="#22c55e" sub="stable"           icon={<CheckCircle size={18} />} />
        <StatCard label="Villages (PSU)"  value={psuData.length}      color="#0ea5e9" sub="covered"          icon={<MapPin size={18} />} />
      </div>

      {/* Embedded Databricks AI/BI Dashboard */}
      <DatabricksDashboardEmbed />

      {/* Cards ranked by complication → home delivery → immunization */}
      <div className="dash-cards-section">
        <div className="dash-cards-header">
          <h2 className="section-title">
            {cardFilter === 'High Priority'
              ? <><AlertTriangle size={18} color="#ef4444" /> Critical Women</>
              : cardFilter === 'Routine Care'
              ? <><AlertCircle size={18} color="#f97316" /> Routine Care Women</>
              : <><Eye size={18} color="#6366f1" /> All Patients</>}
            <span className="count-pill">{displayCards.length}</span>
          </h2>
          <div className="filter-tabs">
            {(['High Priority', 'Routine Care', 'All'] as Filter[]).map(f => (
              <button
                key={f}
                className={`filter-tab ${cardFilter === f ? 'active' : ''}`}
                onClick={() => setCardFilter(f)}
                style={cardFilter === f ? { background: priorityColor(f === 'All' ? 'Medically Fit' : f), color:'#fff', border:'none' } : {}}
              >
                {f === 'High Priority' ? <><AlertTriangle size={13} /> High Priority</>
                  : f === 'Routine Care' ? <><AlertCircle size={13} /> Routine Care</>
                  : <><Eye size={13} /> All</>}
              </button>
            ))}
          </div>
        </div>

        {patients.length === 0 ? (
          <div className="card empty-card">No patients registered yet. Go to the <strong>Patients</strong> tab to add your first patient.</div>
        ) : displayCards.length === 0 ? (
          <div className="card empty-card">No {cardFilter.toLowerCase()} patients found.</div>
        ) : (
          <div className="patient-cards-grid">
            {displayCards.map((p, i) => (
              <PatientCard key={p.patient_id} patient={p} rank={i + 1} onClick={onSelectPatient} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
