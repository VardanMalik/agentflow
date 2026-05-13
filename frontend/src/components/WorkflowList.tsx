import { useEffect, useState, useCallback } from 'react'
import {
  Play, X, RefreshCw, Eye, ChevronLeft, ChevronRight,
  GitBranch, Filter, Search,
} from 'lucide-react'
import { api, type Workflow } from '../api/client'
import { toArray, toTotal } from '../api/normalize'
import StatusBadge from './common/StatusBadge'
import { FullPageSpinner } from './common/LoadingSpinner'
import ErrorMessage from './common/ErrorMessage'

const PAGE_SIZE = 15

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'pending', label: 'Pending' },
  { value: 'cancelled', label: 'Cancelled' },
]

function formatDuration(ms?: number) {
  if (!ms) return '—'
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`
}

function formatDate(ts: string) {
  const d = new Date(ts)
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

interface WorkflowListProps {
  onSelectWorkflow: (id: string) => void
}

export default function WorkflowList({ onSelectWorkflow }: WorkflowListProps) {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [status, setStatus] = useState('')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getWorkflows({
        page,
        page_size: PAGE_SIZE,
        status: status || undefined,
      })
      const list = toArray<Workflow>(data)
      setWorkflows(list)
      setTotal(toTotal(data, list.length))
    } catch (e) {
      setError((e as Error).message)
      setWorkflows([])
    } finally {
      setLoading(false)
    }
  }, [page, status])

  useEffect(() => { load() }, [load])

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const handleAction = async (
    action: 'execute' | 'cancel' | 'retry',
    id: string,
  ) => {
    setActionLoading(`${action}-${id}`)
    try {
      if (action === 'execute') await api.executeWorkflow(id)
      else if (action === 'cancel') await api.cancelWorkflow(id)
      else await api.retryWorkflow(id)
      await load()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setActionLoading(null)
    }
  }

  const safeWorkflows = Array.isArray(workflows) ? workflows : []
  const filtered = search.trim()
    ? safeWorkflows.filter((w) =>
        (w?.name ?? '').toLowerCase().includes(search.toLowerCase()) ||
        (w?.id ?? '').toLowerCase().includes(search.toLowerCase()),
      )
    : safeWorkflows

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Workflows</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {(total ?? 0).toLocaleString()} total workflows
          </p>
        </div>
        <button onClick={load} className="btn-ghost" disabled={loading}>
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
          <input
            type="text"
            placeholder="Search workflows..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="input-field w-full pl-9"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          <select
            value={status}
            onChange={(e) => { setStatus(e.target.value); setPage(1) }}
            className="input-field"
          >
            {STATUS_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </div>

      {error && <ErrorMessage message={error} onRetry={load} compact />}

      {/* Table */}
      <div className="glass rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-left">
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Name
              </th>
              <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Steps
              </th>
              <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider hidden md:table-cell">
                Created
              </th>
              <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider hidden lg:table-cell">
                Duration
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider text-right">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {loading && safeWorkflows.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-16">
                  <FullPageSpinner />
                </td>
              </tr>
            ) : (filtered ?? []).length === 0 ? (
              <tr>
                <td colSpan={6}>
                  <div className="flex flex-col items-center justify-center py-16 gap-3">
                    <GitBranch className="w-10 h-10 text-slate-700" />
                    <p className="text-slate-500">No workflows found</p>
                  </div>
                </td>
              </tr>
            ) : (
              (filtered ?? []).map((wf, idx) => {
                const wfId = wf?.id ?? `wf-${idx}`
                const wfStatus = wf?.status ?? 'pending'
                const steps = Array.isArray(wf?.steps) ? wf.steps : []
                const completedSteps = steps.filter((s) => s?.status === 'completed').length
                return (
                  <tr
                    key={wfId}
                    className="border-b border-slate-800/60 last:border-0 table-row-hover group"
                  >
                    <td className="px-5 py-3.5">
                      <div>
                        <button
                          onClick={() => onSelectWorkflow(wfId)}
                          className="font-medium text-slate-200 hover:text-brand-400 transition-colors text-left"
                        >
                          {wf?.name ?? 'Untitled workflow'}
                        </button>
                        <p className="text-xs text-slate-600 font-mono mt-0.5 truncate max-w-[200px]">
                          {wfId}
                        </p>
                      </div>
                    </td>
                    <td className="px-4 py-3.5">
                      <StatusBadge status={wfStatus} />
                    </td>
                    <td className="px-4 py-3.5">
                      <span className="text-slate-400 tabular-nums">
                        {completedSteps}/{steps.length}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 hidden md:table-cell">
                      <span className="text-slate-500 text-xs">
                        {wf?.created_at ? formatDate(wf.created_at) : '—'}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 hidden lg:table-cell">
                      <span className="text-slate-400 font-mono text-xs">
                        {formatDuration(wf?.duration_ms)}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          title="View details"
                          onClick={() => onSelectWorkflow(wfId)}
                          className="btn-ghost p-1.5"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        {(wfStatus === 'pending' || wfStatus === 'failed') && (
                          <button
                            title="Execute"
                            onClick={() => handleAction('execute', wfId)}
                            disabled={actionLoading === `execute-${wfId}`}
                            className="btn-ghost p-1.5 text-emerald-400 hover:text-emerald-300 hover:bg-emerald-400/10"
                          >
                            <Play className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {wfStatus === 'running' && (
                          <button
                            title="Cancel"
                            onClick={() => handleAction('cancel', wfId)}
                            disabled={actionLoading === `cancel-${wfId}`}
                            className="btn-danger p-1.5"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {wfStatus === 'failed' && (
                          <button
                            title="Retry"
                            onClick={() => handleAction('retry', wfId)}
                            disabled={actionLoading === `retry-${wfId}`}
                            className="btn-ghost p-1.5 text-yellow-400 hover:text-yellow-300 hover:bg-yellow-400/10"
                          >
                            <RefreshCw className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-5 py-3 border-t border-slate-800">
            <p className="text-xs text-slate-500">
              Page {page} of {totalPages} &mdash; {total ?? 0} workflows
            </p>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="btn-ghost p-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                const p = Math.max(1, Math.min(page - 2, totalPages - 4)) + i
                return (
                  <button
                    key={p}
                    onClick={() => setPage(p)}
                    className={`w-8 h-8 rounded-lg text-xs font-medium transition-colors ${
                      p === page
                        ? 'bg-brand-600 text-white'
                        : 'text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {p}
                  </button>
                )
              })}
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="btn-ghost p-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
