import { useEffect, useState, useCallback } from 'react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
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
  type HealthStatus,
} from '../api/client'
import { toArray } from '../api/normalize'
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

interface SectionErrors {
  stats: string | null
  activity: string | null
  health: string | null
}

const NO_ERRORS: SectionErrors = {
  stats: null,
  activity: null,
  health: null,
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [activity, setActivity] = useState<RecentActivity[]>([])
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [errors, setErrors] = useState<SectionErrors>(NO_ERRORS)
  const [loadedOnce, setLoadedOnce] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    const [s, a, h] = await Promise.allSettled([
      api.getStats(),
      api.getRecentActivity(8),
      api.getHealth(),
    ])

    const next: SectionErrors = { ...NO_ERRORS }
    if (s.status === 'fulfilled') setStats(s.value)
    else next.stats = s.reason?.message ?? 'Failed to load stats'
    if (a.status === 'fulfilled') setActivity(toArray<RecentActivity>(a.value))
    else next.activity = a.reason?.message ?? 'Failed to load recent activity'
    if (h.status === 'fulfilled') setHealth(h.value)
    else next.health = h.reason?.message ?? 'Failed to load health'

    setErrors(next)
    setLoading(false)
    setLoadedOnce(true)
  }, [])

  useEffect(() => {
    load()
    const interval = setInterval(load, 30_000)
    return () => clearInterval(interval)
  }, [load])

  if (loading && !loadedOnce) return <FullPageSpinner label="Loading dashboard..." />

  const totalWorkflows = stats?.workflows?.total ?? 0
  const runningWorkflows = stats?.workflows?.running ?? 0
  const failedWorkflows = stats?.workflows?.failed ?? 0
  const successRate = Math.round(stats?.workflows?.success_rate_pct ?? 0)
  const activeAgents = stats?.agents?.active ?? 0
  const avgDurationMs = stats?.performance?.avg_workflow_duration_ms ?? 0

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

      {/* Stats Grid */}
      {errors.stats && !stats ? (
        <ErrorMessage message={errors.stats} onRetry={load} compact />
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatsCard
            title="Total Workflows"
            value={totalWorkflows.toLocaleString()}
            icon={GitBranch}
            accent="blue"
            subtitle={`${runningWorkflows} running`}
          />
          <StatsCard
            title="Success Rate"
            value={`${successRate}%`}
            icon={CheckCircle2}
            accent="green"
            subtitle={`${failedWorkflows} failed`}
          />
          <StatsCard
            title="Active Agents"
            value={activeAgents}
            icon={Cpu}
            accent="violet"
          />
          <StatsCard
            title="Avg Duration"
            value={formatDuration(avgDurationMs)}
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
          <div className="flex items-center justify-center h-[220px]">
            <p className="text-sm text-slate-500">Throughput metrics coming soon</p>
          </div>
        </div>

        {/* System Health */}
        {health ? (
          <div className="glass rounded-xl p-5">
            <h2 className="section-header mb-5">
              <Shield className="w-4 h-4 text-brand-400" />
              System Health
            </h2>
            <div className="space-y-4">
              {/* Overall status */}
              <div className="flex items-center justify-between py-2.5 border-b border-slate-800">
                <span className="text-sm text-slate-400">Overall Status</span>
                <StatusBadge status={health?.status ?? 'unknown'} />
              </div>

              {/* Circuit Breaker */}
              <div className="flex items-center justify-between py-2.5 border-b border-slate-800">
                <div>
                  <p className="text-sm text-slate-300 font-medium">Circuit Breaker</p>
                </div>
                <StatusBadge status={health?.circuit_breaker ?? 'unknown'} />
              </div>

              {/* Bulkhead */}
              <div className="py-2.5 border-b border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-slate-300 font-medium">Bulkhead</p>
                  <span className="text-xs text-slate-500">
                    {health?.bulkhead?.active ?? 0}/{health?.bulkhead?.max ?? 0} slots
                  </span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-brand-500 to-violet-500 rounded-full transition-all"
                    style={{
                      width: `${
                        (health?.bulkhead?.max ?? 0) > 0
                          ? Math.min(100, ((health?.bulkhead?.active ?? 0) / (health?.bulkhead?.max ?? 1)) * 100)
                          : 0
                      }%`,
                    }}
                  />
                </div>
              </div>

              {/* DLQ */}
              <div className="flex items-center justify-between py-2.5 border-b border-slate-800">
                <p className="text-sm text-slate-300 font-medium">Dead Letter Queue</p>
                <span
                  className={`text-sm font-semibold tabular-nums ${
                    (health?.dlq_size ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}
                >
                  {health?.dlq_size ?? 0}
                </span>
              </div>

              {/* Uptime */}
              <div className="flex items-center justify-between pt-1">
                <p className="text-sm text-slate-400">Uptime</p>
                <span className="text-sm text-slate-300 font-mono">
                  {Math.floor((health?.uptime_seconds ?? 0) / 3600)}h{' '}
                  {Math.floor(((health?.uptime_seconds ?? 0) % 3600) / 60)}m
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="glass rounded-xl p-5">
            <h2 className="section-header mb-5">
              <Shield className="w-4 h-4 text-brand-400" />
              System Health
            </h2>
            <div className="flex items-center justify-center h-[220px]">
              <p className="text-sm text-slate-500">
                {errors.health ?? 'Health data unavailable'}
              </p>
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
            {!Array.isArray(activity) || activity.length === 0 ? (
              <p className="text-sm text-slate-500 py-8 text-center">
                {errors.activity ?? 'No recent activity'}
              </p>
            ) : (
              (activity ?? []).map((item, idx) => {
                const ts = item?.created_at ?? item?.timestamp ?? null
                return (
                  <div
                    key={`${item?.id ?? idx}-${idx}`}
                    className="flex items-center gap-3 py-3 border-b border-slate-800/60 last:border-0 table-row-hover -mx-2 px-2 rounded-lg"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-slate-200 font-medium truncate">
                        {item?.name ?? item?.workflow_name ?? 'Unknown workflow'}
                      </p>
                      <p className="text-xs text-slate-500 mt-0.5 capitalize">
                        {item?.event ?? item?.status ?? ''}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <StatusBadge status={item?.status ?? 'unknown'} size="sm" />
                      <span className="text-xs text-slate-600 font-mono">
                        {ts ? formatTime(ts) : ''}
                      </span>
                    </div>
                  </div>
                )
              })
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
      const list = toArray<{
        type?: string
        success_count?: number
        failure_count?: number
        avg_duration_ms?: number
      }>(agents)
      setData(
        list.slice(0, 6).map((a) => ({
          type: (a?.type ?? 'unknown').replace(/_agent$/, '').replace(/_/g, ' '),
          success: a?.success_count ?? 0,
          failure: a?.failure_count ?? 0,
          avg_ms: Math.round(a?.avg_duration_ms ?? 0),
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
