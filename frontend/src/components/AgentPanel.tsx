import { useEffect, useState, useCallback } from 'react'
import {
  Cpu, CheckCircle2, XCircle, BarChart2, Zap, RefreshCw, TrendingUp,
} from 'lucide-react'
import { api, type AgentType } from '../api/client'
import { toArray } from '../api/normalize'
import { FullPageSpinner } from './common/LoadingSpinner'
import ErrorMessage from './common/ErrorMessage'

function formatDuration(ms?: number) {
  const value = ms ?? 0
  if (!value) return '—'
  if (value < 1000) return `${Math.round(value)}ms`
  return `${(value / 1000).toFixed(2)}s`
}

function formatTokens(n?: number) {
  const value = n ?? 0
  if (value < 1000) return value.toString()
  if (value < 1_000_000) return `${(value / 1000).toFixed(1)}k`
  return `${(value / 1_000_000).toFixed(2)}M`
}

function successRate(agent: AgentType) {
  const success = agent?.success_count ?? 0
  const failure = agent?.failure_count ?? 0
  const total = success + failure
  if (!total) return null
  return Math.round((success / total) * 100)
}

interface AgentCardProps {
  agent: AgentType
}

function AgentCard({ agent }: AgentCardProps) {
  const rate = successRate(agent)
  const successCount = agent?.success_count ?? 0
  const failureCount = agent?.failure_count ?? 0
  const executionCount = agent?.execution_count ?? 0
  const totalTokens = agent?.total_tokens ?? 0
  const total = successCount + failureCount
  const successPct = total ? (successCount / total) * 100 : 0

  const rateColor =
    rate === null ? 'text-slate-500'
    : rate >= 90 ? 'text-emerald-400'
    : rate >= 70 ? 'text-yellow-400'
    : 'text-red-400'

  const barColor =
    rate === null ? 'bg-slate-700'
    : rate >= 90 ? 'bg-emerald-500'
    : rate >= 70 ? 'bg-yellow-500'
    : 'bg-red-500'

  const rawType = agent?.type ?? 'unknown'
  const typeName = rawType
    .replace(/_agent$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())

  const isActive = executionCount > 0

  return (
    <div className="glass rounded-xl p-5 hover:bg-slate-800/40 transition-colors group">
      <div className="flex items-start justify-between mb-4">
        <div className="p-2.5 rounded-lg bg-brand-400/10">
          <Cpu className="w-5 h-5 text-brand-400" />
        </div>
        <div className={`flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full ${
          isActive
            ? 'text-emerald-400 bg-emerald-400/10 border border-emerald-500/20'
            : 'text-slate-500 bg-slate-800 border border-slate-700'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-emerald-400' : 'bg-slate-600'}`} />
          {isActive ? 'Active' : 'Idle'}
        </div>
      </div>

      <h3 className="font-semibold text-slate-200 mb-0.5">{typeName}</h3>
      <p className="text-xs text-slate-500 font-mono mb-4">{rawType}</p>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-slate-950/40 rounded-lg p-2.5">
          <p className="text-xs text-slate-500 mb-0.5">Executions</p>
          <p className="text-sm font-semibold text-slate-200 tabular-nums">
            {executionCount.toLocaleString()}
          </p>
        </div>
        <div className="bg-slate-950/40 rounded-lg p-2.5">
          <p className="text-xs text-slate-500 mb-0.5">Avg Time</p>
          <p className="text-sm font-semibold text-slate-200 font-mono">
            {formatDuration(agent?.avg_duration_ms)}
          </p>
        </div>
        <div className="bg-slate-950/40 rounded-lg p-2.5 flex items-center gap-2">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <div>
            <p className="text-xs text-slate-500">Success</p>
            <p className="text-sm font-semibold text-emerald-400 tabular-nums">
              {successCount.toLocaleString()}
            </p>
          </div>
        </div>
        <div className="bg-slate-950/40 rounded-lg p-2.5 flex items-center gap-2">
          <XCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
          <div>
            <p className="text-xs text-slate-500">Failed</p>
            <p className="text-sm font-semibold text-red-400 tabular-nums">
              {failureCount.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Success rate bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Success Rate</span>
          <span className={`text-xs font-semibold ${rateColor}`}>
            {rate !== null ? `${rate}%` : '—'}
          </span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${barColor}`}
            style={{ width: `${successPct}%` }}
          />
        </div>
      </div>

      {/* Token usage */}
      {totalTokens > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-800/60 flex items-center justify-between">
          <span className="text-xs text-slate-500 flex items-center gap-1">
            <Zap className="w-3 h-3" />
            Token Usage
          </span>
          <span className="text-xs text-slate-400 font-mono">
            {formatTokens(totalTokens)}
          </span>
        </div>
      )}
    </div>
  )
}

export default function AgentPanel() {
  const [agents, setAgents] = useState<AgentType[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getAgentTypes()
      setAgents(toArray<AgentType>(data))
    } catch (e) {
      setError((e as Error).message)
      setAgents([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const safeAgents = Array.isArray(agents) ? agents : []
  const totalExecutions = safeAgents.reduce((s, a) => s + (a?.execution_count ?? 0), 0)
  const totalTokens = safeAgents.reduce((s, a) => s + (a?.total_tokens ?? 0), 0)
  const overallSuccess = safeAgents.reduce((s, a) => s + (a?.success_count ?? 0), 0)
  const overallTotal = safeAgents.reduce(
    (s, a) => s + (a?.success_count ?? 0) + (a?.failure_count ?? 0),
    0,
  )
  const overallRate = overallTotal ? Math.round((overallSuccess / overallTotal) * 100) : null

  if (loading && safeAgents.length === 0) return <FullPageSpinner label="Loading agents..." />

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Agents</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {safeAgents.length} agent {safeAgents.length === 1 ? 'type' : 'types'} registered
          </p>
        </div>
        <button onClick={load} className="btn-ghost" disabled={loading}>
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {error && <ErrorMessage message={error} onRetry={load} compact />}

      {/* Summary row */}
      {safeAgents.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-1">
              <BarChart2 className="w-4 h-4 text-brand-400" />
              <p className="text-xs text-slate-500">Total Executions</p>
            </div>
            <p className="text-2xl font-bold text-slate-100 tabular-nums">
              {totalExecutions.toLocaleString()}
            </p>
          </div>
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <p className="text-xs text-slate-500">Overall Success Rate</p>
            </div>
            <p className={`text-2xl font-bold tabular-nums ${
              overallRate !== null
                ? overallRate >= 90 ? 'text-emerald-400'
                : overallRate >= 70 ? 'text-yellow-400'
                : 'text-red-400'
                : 'text-slate-500'
            }`}>
              {overallRate !== null ? `${overallRate}%` : '—'}
            </p>
          </div>
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="w-4 h-4 text-violet-400" />
              <p className="text-xs text-slate-500">Total Tokens Used</p>
            </div>
            <p className="text-2xl font-bold text-slate-100 tabular-nums">
              {formatTokens(totalTokens)}
            </p>
          </div>
        </div>
      )}

      {/* Agent Grid */}
      {safeAgents.length === 0 ? (
        <div className="flex flex-col items-center justify-center min-h-[300px] gap-4">
          <div className="p-4 rounded-full bg-slate-800/60 border border-slate-700">
            <Cpu className="w-8 h-8 text-slate-600" />
          </div>
          <p className="text-slate-500">No agent types registered</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {safeAgents.map((agent, idx) => (
            <AgentCard key={agent?.type ?? `agent-${idx}`} agent={agent} />
          ))}
        </div>
      )}
    </div>
  )
}
