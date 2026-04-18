import { useEffect, useRef, useState } from 'react'

const INSTANCE_URL = 'https://dbc-bc7909d8-379d.cloud.databricks.com'
const WORKSPACE_ID = '7474645411427680'
const DASHBOARD_ID = '01f13b0775f71d64a0e4798da5bd4aa3'
const SDK_CDN      = 'https://cdn.jsdelivr.net/npm/@databricks/aibi-client@0.0.0-alpha.7/+esm'

export default function DatabricksDashboardEmbed() {
  const containerRef                = useRef<HTMLDivElement>(null)
  const [error, setError]           = useState<string | null>(null)
  const [loading, setLoading]       = useState(true)

  useEffect(() => {
    let cancelled = false

    async function init() {
      try {
        // Fetch scoped token from backend
        const res = await fetch('/api/dashboard/embed-token')
        if (!res.ok) {
          const err = await res.json().catch(() => ({}))
          throw new Error(err.detail || `Token fetch failed (${res.status})`)
        }
        const { token } = await res.json()
        if (cancelled || !containerRef.current) return

        // Load SDK from CDN dynamically
        const { DatabricksDashboard } = await import(/* @vite-ignore */ SDK_CDN)

        const dashboard = new DatabricksDashboard({
          instanceUrl: INSTANCE_URL,
          workspaceId: WORKSPACE_ID,
          dashboardId: DASHBOARD_ID,
          token,
          container: containerRef.current,
        })

        await dashboard.initialize()
        if (!cancelled) setLoading(false)
      } catch (e: any) {
        if (!cancelled) {
          setError(e?.message ?? 'Failed to load dashboard')
          setLoading(false)
        }
      }
    }

    init()
    return () => { cancelled = true }
  }, [])

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden', minHeight: 600 }}>
      {loading && !error && (
        <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:600, color:'#6b7280', gap:10 }}>
          <div className="spinner" /> Loading analytics dashboard…
        </div>
      )}
      {error && (
        <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:600, gap:12 }}>
          <span style={{ color:'#dc2626', fontSize:14 }}>⚠ {error}</span>
          <a
            href={`${INSTANCE_URL}/dashboardsv3/${DASHBOARD_ID}/published?o=${WORKSPACE_ID}`}
            target="_blank" rel="noopener noreferrer"
            className="btn-secondary"
            style={{ textDecoration:'none', fontSize:13 }}
          >
            Open in Databricks ↗
          </a>
        </div>
      )}
      <div
        ref={containerRef}
        style={{ width:'100%', height: loading || error ? 0 : 700 }}
      />
    </div>
  )
}
