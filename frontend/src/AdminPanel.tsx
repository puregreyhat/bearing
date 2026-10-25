import { useEffect, useMemo, useState } from "react"
import { ArrowLeft, Database, RefreshCw, Search } from "lucide-react"
import { getAdminTests, type AdminTestLog } from "./lib/api"

const modelNames = { rf: "Random Forest", svm: "SVM", xgboost: "XGBoost" } as const

export default function AdminPanel() {
  const [logs, setLogs] = useState<AdminTestLog[]>([])
  const [search, setSearch] = useState("")
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const loadLogs = async () => {
    setLoading(true)
    setError("")
    try {
      const response = await getAdminTests(search)
      setLogs(response.items)
      setTotal(response.total)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load test logs")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void loadLogs() }, [])

  const completedResults = useMemo(() => new Set(logs.map(log => log.predicted_result)).size, [logs])

  return <div className="app-shell"><main className="main-content admin-main">
    <header className="topbar"><div><button className="admin-back" onClick={() => { window.location.href = "/" }}><ArrowLeft size={15} /> Dashboard</button><h1>Admin panel</h1><p className="admin-subtitle">Diagnosis test history stored in SQLite.</p></div><div className="top-actions"><button className="button header-button" onClick={() => void loadLogs}><RefreshCw size={15} /> Refresh</button><span className="badge badge-green"><span className="status-dot" /> System online</span></div></header>
    <section className="admin-summary"><div className="admin-summary-card"><span>Total tests</span><strong>{total}</strong></div><div className="admin-summary-card"><span>Visible records</span><strong>{logs.length}</strong></div><div className="admin-summary-card"><span>Result classes</span><strong>{completedResults}</strong></div></section>
    <section className="card admin-card"><div className="admin-card-heading"><div><span className="step">LOGS</span><h2>Diagnosis test logs</h2><p>Five key fields are retained for every completed inference.</p></div><div className="admin-search"><Search size={16} /><input value={search} onChange={event => setSearch(event.target.value)} onKeyDown={event => { if (event.key === "Enter") void loadLogs() }} placeholder="Search test ID, file, or result" /><button onClick={() => void loadLogs}>Search</button></div></div>
      {error && <div className="api-error">{error}</div>}
      <div className="admin-table-wrap">{loading ? <div className="admin-empty">Loading test logs…</div> : logs.length === 0 ? <div className="admin-empty"><Database size={20} /><span>No diagnosis logs found.</span></div> : <table className="admin-table"><thead><tr><th>Test ID</th><th>Created</th><th>Input file</th><th>Model</th><th>Result</th></tr></thead><tbody>{logs.map(log => <tr key={log.test_id}><td><code>{log.test_id}</code></td><td>{new Date(log.created_at).toLocaleString()}</td><td title={log.input_filename}>{log.input_filename}</td><td>{modelNames[log.model_family]}</td><td><span className="result-pill">{log.predicted_result.replaceAll("_", " ")}</span></td></tr>)}</tbody></table>}</div>
    </section>
  </main></div>
}
