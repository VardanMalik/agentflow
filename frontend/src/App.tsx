import { useState, useEffect } from 'react'
import {
  LayoutDashboard, GitBranch, Cpu, AlertTriangle, Settings,
  Menu, X, Zap, Circle,
} from 'lucide-react'
import Dashboard from './components/Dashboard'
import WorkflowList from './components/WorkflowList'
import WorkflowDetail from './components/WorkflowDetail'
import AgentPanel from './components/AgentPanel'
import DLQPanel from './components/DLQPanel'
import SettingsPanel from './components/SettingsPanel'
import { api } from './api/client'

type View =
  | 'dashboard'
  | 'workflows'
  | 'workflow-detail'
  | 'agents'
  | 'dlq'
  | 'settings'

interface NavItem {
  id: View
  label: string
  icon: React.ElementType
  badge?: number
}

const NAV_ITEMS: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'workflows', label: 'Workflows', icon: GitBranch },
  { id: 'agents',    label: 'Agents',    icon: Cpu },
  { id: 'dlq',       label: 'Dead Letter Queue', icon: AlertTriangle },
  { id: 'settings',  label: 'Settings',  icon: Settings },
]

export default function App() {
  const [view, setView] = useState<View>('dashboard')
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'degraded' | 'unhealthy'>('healthy')
  const [dlqCount, setDlqCount] = useState(0)

  useEffect(() => {
    const check = async () => {
      try {
        const [health, stats] = await Promise.all([api.getHealth(), api.getStats()])
        setHealthStatus(
          health.status === 'healthy' ? 'healthy'
          : health.status === 'degraded' ? 'degraded'
          : 'unhealthy',
        )
        setDlqCount(stats.dlq_size)
      } catch {
        setHealthStatus('unhealthy')
      }
    }
    check()
    const interval = setInterval(check, 30_000)
    return () => clearInterval(interval)
  }, [])

  const navigate = (v: View) => {
    setView(v)
    setSidebarOpen(false)
    if (v !== 'workflow-detail') setSelectedWorkflowId(null)
  }

  const openWorkflow = (id: string) => {
    setSelectedWorkflowId(id)
    setView('workflow-detail')
    setSidebarOpen(false)
  }

  const healthColor =
    healthStatus === 'healthy' ? 'text-emerald-400 bg-emerald-400/10'
    : healthStatus === 'degraded' ? 'text-yellow-400 bg-yellow-400/10'
    : 'text-red-400 bg-red-400/10 animate-pulse'

  const healthDot =
    healthStatus === 'healthy' ? 'bg-emerald-400'
    : healthStatus === 'degraded' ? 'bg-yellow-400'
    : 'bg-red-400'

  const activeView = view === 'workflow-detail' ? 'workflows' : view

  const navItemsWithBadges: NavItem[] = NAV_ITEMS.map((item) =>
    item.id === 'dlq' && dlqCount > 0 ? { ...item, badge: dlqCount } : item,
  )

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950 bg-grid-pattern">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-64 flex flex-col
          bg-slate-900/95 backdrop-blur-xl border-r border-slate-800/60
          transform transition-transform duration-300
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:relative lg:translate-x-0`}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 h-16 border-b border-slate-800/60 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="font-bold text-slate-100 text-sm leading-none">AgentFlow</p>
            <p className="text-[10px] text-slate-500 mt-0.5 leading-none">Orchestration Platform</p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="ml-auto lg:hidden text-slate-500 hover:text-slate-300"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
          <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-widest px-3 py-2">
            Navigation
          </p>
          {navItemsWithBadges.map((item) => {
            const Icon = item.icon
            const isActive = item.id === activeView
            return (
              <button
                key={item.id}
                onClick={() => navigate(item.id)}
                className={`nav-item w-full ${isActive ? 'active' : ''}`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="flex-1 text-left">{item.label}</span>
                {item.badge !== undefined && (
                  <span className="ml-auto badge bg-red-500/10 text-red-400 border border-red-500/20">
                    {item.badge > 99 ? '99+' : item.badge}
                  </span>
                )}
              </button>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-3 border-t border-slate-800/60 shrink-0">
          <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${healthColor}`}>
            <Circle className={`w-2 h-2 rounded-full ${healthDot} fill-current`} />
            <span className="capitalize font-medium">System {healthStatus}</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="h-16 flex items-center justify-between px-5 border-b border-slate-800/60 bg-slate-900/60 backdrop-blur-sm shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden text-slate-400 hover:text-slate-200"
          >
            <Menu className="w-5 h-5" />
          </button>

          {/* Breadcrumb / page title */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-500">AgentFlow</span>
            <span className="text-slate-700">/</span>
            <span className="text-slate-300 font-medium capitalize">
              {view === 'workflow-detail' && selectedWorkflowId
                ? `Workflow · ${selectedWorkflowId.slice(0, 8)}…`
                : view.replace('-', ' ')}
            </span>
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-3">
            {dlqCount > 0 && (
              <button
                onClick={() => navigate('dlq')}
                className="flex items-center gap-1.5 text-xs text-red-400 bg-red-400/10 border border-red-500/20 px-2.5 py-1.5 rounded-lg hover:bg-red-400/20 transition-colors"
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                {dlqCount} in DLQ
              </button>
            )}
            <div className={`hidden sm:flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg ${healthColor}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${healthDot} ${healthStatus !== 'healthy' ? 'animate-pulse' : ''}`} />
              <span className="capitalize">{healthStatus}</span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-5 lg:p-8">
          {view === 'dashboard' && <Dashboard />}
          {view === 'workflows' && (
            <WorkflowList onSelectWorkflow={openWorkflow} />
          )}
          {view === 'workflow-detail' && selectedWorkflowId && (
            <WorkflowDetail
              workflowId={selectedWorkflowId}
              onBack={() => navigate('workflows')}
            />
          )}
          {view === 'agents' && <AgentPanel />}
          {view === 'dlq' && <DLQPanel />}
          {view === 'settings' && <SettingsPanel />}
        </main>
      </div>
    </div>
  )
}
