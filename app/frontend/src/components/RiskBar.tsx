interface Props {
  label: string
  value: number | null
  color?: string
}

export default function RiskBar({ label, value, color }: Props) {
  const pct = Math.round((value ?? 0) * 100)
  const c   = color ?? (pct >= 60 ? '#dc2626' : pct >= 30 ? '#d97706' : '#059669')

  return (
    <div className="risk-bar-row">
      <span className="risk-bar-label">{label}</span>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width:`${pct}%`, background:c }} />
      </div>
      <span className="risk-bar-pct" style={{ color: c }}>{value == null ? '—' : `${pct}%`}</span>
    </div>
  )
}
