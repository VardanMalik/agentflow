import { useEffect, useState, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend,
} from 'recharts'
import {
  GitBranch, CheckCircle2, Cpu, Timer, Shield, AlertTriangle,
  Activity, RefreshCw, Layers,
} from 'lucide-react'
import {
  api,
  type DashboardStats,
  type RecentActivity,
  type ThroughputPoint,
  type HealthStatus,
} from '../api/client'
import StatsCard from './common/StatsCard'
import StatusBadge from './common/StatusBadge'
import { FullPageSpinner } from './common/LoadingSpinner'
import ErrorMessage from './common/ErrorMessage'

function formatDuration(ms?: number) {
  if (!ms) return '—'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatTime(ts: string) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#0f172a',
  border: '1px solid #1e293b',
  borderRadius: '8px',
  color: '#cbd5e1',
  fontSize: '12px',
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [activity, setActivity] = useState<RecentActivity[]>([])
  const [throughput, setThroughput] = useState<ThroughputPoint[]>([])
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [s, a, t, h] = await Promise.all([
        api.getStats(),
        api.getRecentActivity(8),
        api.getThroughput(60),
        api.getHealth(),
      ])
      setStats(s)
      setActivity(a)
      setThroughput(t)
      setHealth(h)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    const interval = setInterval(load, 30_000)
    return () => clearInterval(interval)
  }, [load])

  if (loading && !stats) return <FullPageSpinner label="Loading dashboard..." />
  if (error && !stats) return <ErrorMessage message={error} onRetry={load} />

  const successRate = stats ? Math.round(stats.success_rate * 100) : 0

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">System overview and performance metrics</p>
        </div>
        <button onClick={load} className="btn-ghost" disabled={loading}>
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {error && <ErrorMessage message={error} onRetry={load} compact />}

      {/* Stats Grid */}
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatsCard
            title="Total Workflows"
            value={stats.total_workflows.toLocaleString()}
            icon={GitBranch}
            accent="blue"
            subtitle={`${stats.running_workflows} running`}
          />
          <StatsCard
            title="Success Rate"
            value={`${successRate}%`}
            icon={CheckCircle2}
            accent="green"
            subtitle={`${stats.failed_workflows} failed`}
          />
          <StatsCard
            title="Active Agents"
            value={stats.active_agents}
            icon={Cpu}
            accent="violet"
          />
          <StatsCard
            title="Avg Duration"
            value={formatDuration(stats.avg_duration_ms)}
            icon={Timer}
            accent="yellow"
          />
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Throughput Chart */}
        <div className="glass rounded-xl p-5 lg:col-span-2">
          <h2 className="section-header mb-5">
            <Activity className="w-4 h-4 text-brand-400" />
            Workflow Throughput
            <span className="ml-auto text-xs font-normal text-slate-500">Last 60 min</span>
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={throughput} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTime}
                tick={{ fontSize: 11, fill: '#64748b' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748b' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={CHART_TOOLTIP_STYLE}
                labelFormatter={(v) => formatTime(v as string)}
              />
              <Legend
                wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }}
                iconType="circle"
                iconSize={8}
              />
              <Line
                type="monotone"
                dataKey="completed"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#6366f1' }}
              />
              <Line
                type="monotone"
                dataKey="failed"
                stroke="#f87171"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#f87171' }}
              />
              <Line
                type="monotone"
                dataKey="running"
                stroke="#a78bfa"
                strokeWidth={2}
                strokeDasharray="4 2"
                dot={false}
                activeDot={{ r: 4, fill: '#a78bfa' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* System Health */}
        {health && (
          <div className="glass rounded-xl p-5">
            <h2 className="section-header mb-5">
              <Shield className="w-4 h-4 text-brand-400" />
              System Health
            </h2>
            <div className="space-y-4">
              {/* Overall status */}
              <div className="flex items-center justify-between py-2.5 border-b border-slate-800">
                <span className="text-sm text-slate-400">Overall Status</span>
                <StatusBadge status={health.status} />
              </div>

              {/* Circuit Breaker */}
              <div className="flex items-center justify-between py-2.5 border-b border-slate-800">
                <div>
                  <p className="text-sm text-slate-300 font-medium">Circuit Breaker</p>
                </div>
                <StatusBadge status={health.circuit_breaker} />
              </div>

              {/* Bulkhead */}
              <div className="py-2.5 border-b border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-slate-300 font-medium">Bulkhead</p>
                  <span className="text-xs text-slate-500">
                    {health.bulkhead.active}/{health.bulkhead.max} slots
                  </span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-brand-500 to-violet-500 rounded-full transition-all"
                    style={{
                      width: `${Math.min(100, (health.bulkhead.active / health.bulkhead.max) * 100)}%`,
                    }}
                  />
                </div>
              </div>

              {/* DLQ */}
              <div className="flex items-center justify-between py-2.5 border-b border-slate-800">
                <p className="text-sm text-slate-300 font-medium">Dead Letter Queue</p>
                <span
                  className={`text-sm font-semibold tabular-nums ${
                    (stats?.dlq_size ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}
                >
                  {stats?.dlq_size ?? 0}
                </span>
              </div>

              {/* Uptime */}
              <div className="flex items-center justify-between pt-1">
                <p className="text-sm text-slate-400">Uptime</p>
                <span className="text-sm text-slate-300 font-mono">
                  {Math.floor(health.uptime_seconds / 3600)}h{' '}
                  {Math.floor((health.uptime_seconds % 3600) / 60)}m
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Recent Activity + Agent Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Recent Workflows */}
        <div className="glass rounded-xl p-5">
          <h2 className="section-header mb-4">
            <Layers className="w-4 h-4 text-brand-400" />
            Recent Activity
          </h2>
          <div className="space-y-0">
            {activity.length === 0 ? (
              <p className="text-sm text-slate-500 py-8 text-center">No recent activity</p>
            ) : (
              activity.map((item, idx) => (
                <div
                  key={`${item.id}-${idx}`}
                  className="flex items-center gap-3 py-3 border-b border-slate-800/60 last:border-0 table-row-hover -mx-2 px-2 rounded-lg"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-slate-200 font-medium truncate">
                      {item.workflow_name}
                    </p>
                    <p className="text-xs text-slate-500 mt-0.5">{item.event}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <StatusBadge status={item.status} size="sm" />
                    <span className="text-xs text-slate-600 font-mono">
                      {formatTime(item.timestamp)}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Agent Performance */}
        <AgentPerformanceChart />
      </div>
    </div>
  )
}

function AgentPerformanceChart() {
  const [data, setData] = useState<
    { type: string; success: number; failure: number; avg_ms: number }[]
  >([])

  useEffect(() => {
    api.getAgentTypes().then((agents) => {
      setData(
        agents.slice(0, 6).map((a) => ({
          type: a.type.replace(/_agent$/, '').replace(/_/g, ' '),
          success: a.success_count,
          failure: a.failure_count,
          avg_ms: Math.round(a.avg_duration_ms),
        })),
      )
    }).catch(() => {})
  }, [])

  return (
    <div className="glass rounded-xl p-5">
      <h2 className="section-header mb-5">
        <AlertTriangle className="w-4 h-4 text-brand-400" />
        Agent Performance
      </h2>
      {data.length === 0 ? (
        <div className="flex items-center justify-center h-[220px]">
          <p className="text-sm text-slate-500">No agent data available</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis
              dataKey="type"
              tick={{ fontSize: 11, fill: '#64748b' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#64748b' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                border: '1px solid #1e293b',
                borderRadius: '8px',
                color: '#cbd5e1',
                fontSize: '12px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} iconSize={8} />
            <Bar dataKey="success" fill="#6366f1" radius={[3, 3, 0, 0]} name="Success" />
            <Bar dataKey="failure" fill="#f87171" radius={[3, 3, 0, 0]} name="Failed" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
