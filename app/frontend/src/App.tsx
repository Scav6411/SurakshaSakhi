import { useState, useEffect, useCallback } from 'react'
import { LayoutDashboard, Users } from 'lucide-react'
import { Patient, PSURow, Visit, FormState, EMPTY_FORM } from './types'
import { patientToForm } from './utils'
import {
  fetchPatients, fetchPatient, createPatient,
  updatePatient, deletePatient, fetchVisits, fetchPSUSummary,
} from './api/client'
import Dashboard      from './components/Dashboard'
import PatientList    from './components/PatientList'
import PatientDetail  from './components/PatientDetail'
import PatientForm    from './components/PatientForm'
import VoiceAssistant from './components/VoiceAssistant'
import './App.css'

type Tab = 'dashboard' | 'patients'

export default function App() {
  const [tab,        setTab]        = useState<Tab>('dashboard')
  const [patients,   setPatients]   = useState<Patient[]>([])
  const [psuData,    setPsuData]    = useState<PSURow[]>([])
  const [visits,     setVisits]     = useState<Visit[]>([])
  const [selected,   setSelected]   = useState<Patient | null>(null)
  const [showAdd,    setShowAdd]    = useState(false)
  const [editTarget, setEditTarget] = useState<Patient | null>(null)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState('')

  const loadAll = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [p, d] = await Promise.all([fetchPatients(), fetchPSUSummary()])
      setPatients(Array.isArray(p) ? p : [])
      setPsuData(Array.isArray(d) ? d : [])
    } catch {
      setError('Could not reach backend. Is the Databricks App running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  const selectPatient = async (p: Patient) => {
    setSelected(p)
    setTab('patients')
    const v = await fetchVisits(p.patient_id)
    setVisits(Array.isArray(v) ? v : [])
  }

  const handleCreate = async (form: FormState) => {
    await createPatient(form)
    setShowAdd(false)
    await loadAll()
  }

  const handleUpdate = async (form: FormState) => {
    if (!editTarget) return
    await updatePatient(editTarget.patient_id, form)
    setEditTarget(null)
    await loadAll()
    if (selected?.patient_id === editTarget.patient_id) {
      const fresh = await fetchPatient(editTarget.patient_id)
      setSelected(fresh)
      const v = await fetchVisits(editTarget.patient_id)
      setVisits(Array.isArray(v) ? v : [])
    }
  }

  const handleDelete = async (pid: string) => {
    if (!confirm('Remove this patient?')) return
    await deletePatient(pid)
    if (selected?.patient_id === pid) setSelected(null)
    await loadAll()
  }

  const switchTab = (t: Tab) => {
    setTab(t)
    if (t === 'dashboard') setSelected(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1 className="app-title">Suraksha</h1>
          <span className="app-subtitle">Maternal Risk Intelligence — Araria, Bihar</span>
        </div>
        <nav className="tabs">
          <button className={tab === 'dashboard' ? 'tab active' : 'tab'} onClick={() => switchTab('dashboard')}>
            <LayoutDashboard size={15} /> Dashboard
          </button>
          <button className={tab === 'patients' ? 'tab active' : 'tab'} onClick={() => switchTab('patients')}>
            <Users size={15} /> Patients {patients.length > 0 && <span className="tab-badge">{patients.length}</span>}
          </button>
        </nav>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="main">
        {loading ? (
          <div className="loading">Loading data…</div>
        ) : tab === 'dashboard' ? (
          <Dashboard patients={patients} psuData={psuData} onSelectPatient={selectPatient} />
        ) : selected ? (
          <PatientDetail
            patient={selected}
            visits={visits}
            onBack={() => setSelected(null)}
            onEdit={p => setEditTarget(p)}
            onDelete={handleDelete}
          />
        ) : (
          <PatientList
            patients={patients}
            onSelect={selectPatient}
            onEdit={p => setEditTarget(p)}
            onDelete={handleDelete}
            onAdd={() => setShowAdd(true)}
          />
        )}
      </main>

      {showAdd && (
        <PatientForm
          initial={EMPTY_FORM}
          title="Register New Patient"
          onSave={handleCreate}
          onClose={() => setShowAdd(false)}
        />
      )}

      {editTarget && (
        <PatientForm
          initial={patientToForm(editTarget)}
          title={`Update: ${editTarget.name}`}
          onSave={handleUpdate}
          onClose={() => setEditTarget(null)}
        />
      )}

      <VoiceAssistant patients={patients} psuData={psuData} />
    </div>
  )
}
