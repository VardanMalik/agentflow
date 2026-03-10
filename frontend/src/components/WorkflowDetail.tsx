import { useEffect, useState, useCallback } from 'react'
import {
  ArrowLeft, RefreshCw, Play, X, CheckCircle2, XCircle,
  Clock, ChevronDown, ChevronRight, Cpu, Zap, AlertCircle,
  Circle,
} from 'lucide-react'
import { api, useWebSocket, type Workflow, type WorkflowStep } from '../api/client'
import StatusBadge from './common/StatusBadge'
import { FullPageSpinner } from './common/LoadingSpinner'
import ErrorMessage from './common/ErrorMessage'

function formatDuration(ms?: number) {
  if (!ms) return null
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function formatTimestamp(ts?: string) {
  if (!ts) return null
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

// ─── Step Timeline Item ───────────────────────────────────────────────────────

interface StepItemProps {
  step: WorkflowStep
  isLast: boolean
  index: number
}

function StepItem({ step, isLast, index }: StepItemProps) {
  const [expanded, setExpanded] = useState(false)

  const StatusIcon =
    step.status === 'completed' ? CheckCircle2
    : step.status === 'failed' ? XCircle
    : step.status === 'running' ? Zap
    : step.status === 'skipped' ? ChevronRight
    : Circle

  const iconColor =
    step.status === 'completed' ? 'text-emerald-400'
    : step.status === 'failed' ? 'text-red-400'
    : step.status === 'running' ? 'text-brand-400 animate-pulse'
    : 'text-slate-600'

  const lineColor =
    step.status === 'completed' ? 'bg-emerald-500/30'
    : step.status === 'failed' ? 'bg-red-500/30'
    : step.status === 'running' ? 'bg-brand-500/30'
    : 'bg-slate-800'

  const hasDetails = step.output || step.error

  return (
    <div className="flex gap-4">
      {/* Timeline column */}
      <div className="flex flex-col items-center">
        <div className={`p-1.5 rounded-full bg-slate-900 border border-slate-800 z-10 ${iconColor}`}>
          <StatusIcon className="w-4 h-4" />
        </div>
        {!isLast && <div className={`w-0.5 flex-1 mt-1 min-h-[24px] ${lineColor}`} />}
      </div>

      {/* Step content */}
      <div className="flex-1 pb-5 min-w-0">
        <div
          className={`glass rounded-xl p-4 ${hasDetails ? 'cursor-pointer hover:bg-slate-800/40' : ''} transition-colors`}
          onClick={() => hasDetails && setExpanded(!expanded)}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-slate-600">#{index + 1}</span>
                <h3 className="text-sm font-semibold text-slate-200 truncate">{step.name}</h3>
              </div>
              <div className="flex items-center gap-3 mt-1">
                <span className="text-xs text-slate-500 flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  {step.agent_type}
                </span>
                {step.tokens_used && (
                  <span className="text-xs text-slate-600">{step.tokens_used.toLocaleString()} tokens</span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <StatusBadge status={step.status} size="sm" />
              {hasDetails && (
                <ChevronDown
                  className={`w-4 h-4 text-slate-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
                />
              )}
            </div>
          </div>

          {/* Timing info */}
          {(step.started_at || step.duration_ms) && (
            <div className="flex items-center gap-4 mt-3 pt-3 border-t border-slate-800/60">
              {step.started_at && (
                <span className="flex items-center gap-1 text-xs text-slate-500">
                  <Clock className="w-3 h-3" />
                  Started {formatTimestamp(step.started_at)}
                </span>
              )}
              {step.duration_ms && (
                <span className="text-xs text-slate-500 font-mono">
                  {formatDuration(step.duration_ms)}
                </span>
              )}
            </div>
          )}

          {/* Progress bar for running */}
          {step.status === 'running' && (
            <div className="mt-3 h-0.5 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-brand-500 to-violet-500 rounded-full animate-pulse w-2/3" />
            </div>
          )}

          {/* Expanded details */}
          {expanded && (
            <div className="mt-4 pt-4 border-t border-slate-800/60 space-y-3 animate-fade-in">
              {step.error && (
                <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-xs font-semibold text-red-400">Error</span>
                  </div>
                  <p className="text-xs text-red-300/80 font-mono leading-relaxed">
                    {step.error}
                  </p>
                </div>
              )}
              {step.output && (
                <div>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    Output
                  </p>
                  <pre className="text-xs text-slate-400 font-mono bg-slate-950/60 rounded-lg p-3 overflow-auto max-h-48 leading-relaxed">
                    {JSON.stringify(step.output, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

interface WorkflowDetailProps {
  workflowId: string
  onBack: () => void
}

export default function WorkflowDetail({ workflowId, onBack }: WorkflowDetailProps) {
  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const { messages, connected } = useWebSocket(workflowId)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getWorkflow(workflowId)
      setWorkflow(data)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [workflowId])

  useEffect(() => { load() }, [load])

  // Reload on relevant WS messages
  useEffect(() => {
    const last = messages[messages.length - 1]
    if (last?.workflow_id === workflowId) {
      load()
    }
  }, [messages, workflowId, load])

  const handleAction = async (action: 'execute' | 'cancel' | 'retry') => {
    setActionLoading(action)
    try {
      if (action === 'execute') await api.executeWorkflow(workflowId)
      else if (action === 'cancel') await api.cancelWorkflow(workflowId)
      else await api.retryWorkflow(workflowId)
      await load()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setActionLoading(null)
    }
  }

  if (loading && !workflow) return <FullPageSpinner label="Loading workflow..." />
  if (error && !workflow) return <ErrorMessage message={error} onRetry={load} />

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={onBack} className="btn-ghost p-2">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-slate-100 truncate">
            {workflow?.name ?? workflowId}
          </h1>
          <p className="text-xs text-slate-500 font-mono mt-0.5">{workflowId}</p>
        </div>
        <div className="flex items-center gap-2">
          {/* WS indicator */}
          <div
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${
              connected
                ? 'text-emerald-400 bg-emerald-400/10 border-emerald-500/20'
                : 'text-slate-600 bg-slate-800/40 border-slate-700/40'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
            {connected ? 'Live' : 'Offline'}
          </div>
          <button onClick={load} className="btn-ghost" disabled={loading}>
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} onRetry={load} compact />}

      {workflow && (
        <>
          {/* Info Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <InfoCard label="Status">
              <StatusBadge status={workflow.status} />
            </InfoCard>
            <InfoCard label="Steps">
              <span className="text-slate-200 font-semibold">
                {workflow.steps.filter((s) => s.status === 'completed').length} / {workflow.steps.length}
              </span>
            </InfoCard>
            <InfoCard label="Duration">
              <span className="text-slate-200 font-mono text-sm">
                {formatDuration(workflow.duration_ms) ?? '—'}
              </span>
            </InfoCard>
            <InfoCard label="Created">
              <span className="text-slate-400 text-xs">
                {new Date(workflow.created_at).toLocaleString()}
              </span>
            </InfoCard>
          </div>

          {/* Description */}
          {workflow.description && (
            <div className="glass rounded-xl p-4">
              <p className="text-sm text-slate-400">{workflow.description}</p>
            </div>
          )}

          {/* Error banner */}
          {workflow.error && (
            <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-semibold text-red-400 mb-1">Workflow Error</p>
                <p className="text-sm text-red-300/80 font-mono">{workflow.error}</p>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2">
            {(workflow.status === 'pending' || workflow.status === 'failed') && (
              <button
                onClick={() => handleAction('execute')}
                disabled={actionLoading === 'execute'}
                className="btn-primary"
              >
                <Play className="w-4 h-4" />
                Execute
              </button>
            )}
            {workflow.status === 'running' && (
              <button
                onClick={() => handleAction('cancel')}
                disabled={actionLoading === 'cancel'}
                className="btn-danger border border-red-500/20"
              >
                <X className="w-4 h-4" />
                Cancel
              </button>
            )}
            {workflow.status === 'failed' && (
              <button
                onClick={() => handleAction('retry')}
                disabled={actionLoading === 'retry'}
                className="btn-ghost"
              >
                <RefreshCw className="w-4 h-4" />
                Retry
              </button>
            )}
          </div>

          {/* Step Timeline */}
          <div className="glass rounded-xl p-5">
            <h2 className="section-header mb-6">
              Step Timeline
              <span className="ml-auto text-xs font-normal text-slate-500">
                {workflow.steps.length} steps
              </span>
            </h2>
            {workflow.steps.length === 0 ? (
              <p className="text-sm text-slate-500 py-4 text-center">No steps defined</p>
            ) : (
              <div>
                {workflow.steps.map((step, i) => (
                  <StepItem
                    key={step.id}
                    step={step}
                    index={i}
                    isLast={i === workflow.steps.length - 1}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Live events */}
          {messages.length > 0 && (
            <div className="glass rounded-xl p-5">
              <h2 className="section-header mb-4">
                <Zap className="w-4 h-4 text-brand-400" />
                Live Events
                <span className="ml-auto text-xs font-normal text-slate-500">
                  {messages.length} events
                </span>
              </h2>
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {[...messages].reverse().map((msg, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs font-mono">
                    <span className="text-slate-600 shrink-0">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="text-brand-400">{msg.type}</span>
                    {msg.status && <StatusBadge status={msg.status} size="sm" />}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function InfoCard({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="glass rounded-xl p-4">
      <p className="text-xs text-slate-500 mb-1.5">{label}</p>
      {children}
    </div>
  )
}
