import { useState } from 'react'
import { Settings, Server, Wifi, Info, ExternalLink } from 'lucide-react'

export default function SettingsPanel() {
  const [apiUrl, setApiUrl] = useState(
    import.meta.env.VITE_API_URL ?? '/api',
  )
  const [wsUrl, setWsUrl] = useState(
    import.meta.env.VITE_WS_URL ?? `ws://${window.location.host}/ws`,
  )
  const [saved, setSaved] = useState(false)

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault()
    // In a real app these would persist to localStorage / env
    setSaved(true)
    setTimeout(() => setSaved(false), 2500)
  }

  return (
    <div className="space-y-6 max-w-2xl animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Settings</h1>
        <p className="text-sm text-slate-500 mt-0.5">Configure the dashboard connection</p>
      </div>

      <form onSubmit={handleSave} className="space-y-5">
        {/* Connection */}
        <div className="glass rounded-xl p-5 space-y-4">
          <h2 className="section-header">
            <Server className="w-4 h-4 text-brand-400" />
            API Connection
          </h2>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">
              API Base URL
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="input-field w-full"
            />
            <p className="text-xs text-slate-500 mt-1.5">
              Base URL for the AgentFlow REST API
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">
              WebSocket URL
            </label>
            <input
              type="text"
              value={wsUrl}
              onChange={(e) => setWsUrl(e.target.value)}
              placeholder="ws://localhost:8000/ws"
              className="input-field w-full"
            />
            <p className="text-xs text-slate-500 mt-1.5">
              WebSocket endpoint for real-time workflow events
            </p>
          </div>
        </div>

        {/* About */}
        <div className="glass rounded-xl p-5 space-y-3">
          <h2 className="section-header">
            <Info className="w-4 h-4 text-brand-400" />
            About
          </h2>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-slate-500">Version</p>
              <p className="text-slate-300 font-mono">1.0.0</p>
            </div>
            <div>
              <p className="text-slate-500">Stack</p>
              <p className="text-slate-300">React 18 + Vite + Tailwind</p>
            </div>
            <div>
              <p className="text-slate-500">Charts</p>
              <p className="text-slate-300">Recharts</p>
            </div>
            <div>
              <p className="text-slate-500">Real-time</p>
              <p className="text-slate-300">WebSocket</p>
            </div>
          </div>
        </div>

        {/* Connection status */}
        <div className="glass rounded-xl p-5">
          <h2 className="section-header mb-4">
            <Wifi className="w-4 h-4 text-brand-400" />
            Connection Status
          </h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">API Proxy</span>
              <span className="flex items-center gap-1.5 text-xs text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Configured via Vite proxy
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">WebSocket</span>
              <span className="flex items-center gap-1.5 text-xs text-slate-400">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                Auto-connect on view
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button type="submit" className="btn-primary">
            <Settings className="w-4 h-4" />
            Save Settings
          </button>
          {saved && (
            <span className="text-sm text-emerald-400 animate-fade-in">
              Settings saved
            </span>
          )}
          <a
            href="https://github.com/anthropics/claude-code"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-ghost ml-auto"
          >
            <ExternalLink className="w-4 h-4" />
            Docs
          </a>
        </div>
      </form>
    </div>
  )
}
