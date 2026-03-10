import { useEffect, useState, useCallback } from 'react'
import {
  AlertTriangle, RefreshCw, Trash2, RotateCcw, X, ChevronLeft, ChevronRight,
  Clock, Info,
} from 'lucide-react'
import { api, type DLQEntry } from '../api/client'
import { FullPageSpinner } from './common/LoadingSpinner'
import ErrorMessage from './common/ErrorMessage'

const PAGE_SIZE = 20

function formatDate(ts: string) {
  return new Date(ts).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// ─── Detail Modal ─────────────────────────────────────────────────────────────

interface DetailModalProps {
  entry: DLQEntry
  onClose: () => void
  onRetry: () => void
  onPurge: () => void
}

function DetailModal({ entry, onClose, onRetry, onPurge }: DetailModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="glass rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col animate-slide-in shadow-2xl">
        {/* Modal header */}
        <div className="flex items-start justify-between p-5 border-b border-slate-800">
          <div>
            <h3 className="font-semibold text-slate-100">DLQ Entry Details</h3>
            <p className="text-xs text-slate-500 font-mono mt-0.5">{entry.id}</p>
          </div>
          <button onClick={onClose} className="btn-ghost p-1.5">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* Meta */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-950/40 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Workflow ID</p>
              <p className="text-sm text-slate-300 font-mono truncate">{entry.workflow_id}</p>
            </div>
            {entry.step_id && (
              <div className="bg-slate-950/40 rounded-lg p-3">
                <p className="text-xs text-slate-500 mb-1">Step ID</p>
                <p className="text-sm text-slate-300 font-mono truncate">{entry.step_id}</p>
              </div>
            )}
            <div className="bg-slate-950/40 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Retry Count</p>
              <p className="text-sm font-semibold text-yellow-400">{entry.retry_count}</p>
            </div>
            <div className="bg-slate-950/40 rounded-lg p-3">
              <p className="text-xs text-slate-500 mb-1">Created At</p>
              <p className="text-sm text-slate-300">{formatDate(entry.created_at)}</p>
            </div>
          </div>

          {/* Error */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Error
            </p>
            <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3">
              <p className="text-sm text-red-300/90 font-mono leading-relaxed">{entry.error}</p>
            </div>
          </div>

          {/* Payload */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Payload
            </p>
            <pre className="text-xs text-slate-400 font-mono bg-slate-950/60 border border-slate-800 rounded-lg p-3 overflow-auto max-h-48 leading-relaxed">
              {JSON.stringify(entry.payload, null, 2)}
            </pre>
          </div>
        </div>

        {/* Modal footer */}
        <div className="flex items-center justify-end gap-2 p-4 border-t border-slate-800">
          <button onClick={onClose} className="btn-ghost">Close</button>
          <button onClick={onRetry} className="btn-ghost text-yellow-400 hover:bg-yellow-400/10">
            <RotateCcw className="w-4 h-4" />
            Retry
          </button>
          <button onClick={onPurge} className="btn-danger border border-red-500/20">
            <Trash2 className="w-4 h-4" />
            Purge
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function DLQPanel() {
  const [entries, setEntries] = useState<DLQEntry[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<DLQEntry | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [confirmPurgeAll, setConfirmPurgeAll] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getDLQ({ page, page_size: PAGE_SIZE })
      setEntries(data.entries)
      setTotal(data.total)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => { load() }, [load])

  const handleRetry = async (id: string) => {
    setActionLoading(`retry-${id}`)
    try {
      await api.retryDLQEntry(id)
      setSelected(null)
      await load()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setActionLoading(null)
    }
  }

  const handlePurge = async (id: string) => {
    setActionLoading(`purge-${id}`)
    try {
      await api.purgeDLQEntry(id)
      setSelected(null)
      await load()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setActionLoading(null)
    }
  }

  const handlePurgeAll = async () => {
    setActionLoading('purge-all')
    try {
      await api.purgeAllDLQ()
      setConfirmPurgeAll(false)
      await load()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setActionLoading(null)
    }
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Dead Letter Queue</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {total} failed {total === 1 ? 'entry' : 'entries'} awaiting action
          </p>
        </div>
        <div className="flex items-center gap-2">
          {total > 0 && !confirmPurgeAll && (
            <button
              onClick={() => setConfirmPurgeAll(true)}
              className="btn-danger border border-red-500/20"
            >
              <Trash2 className="w-4 h-4" />
              Purge All
            </button>
          )}
          {confirmPurgeAll && (
            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
              <span className="text-sm text-red-400">Confirm purge all?</span>
              <button
                onClick={handlePurgeAll}
                disabled={actionLoading === 'purge-all'}
                className="text-xs font-semibold text-red-400 hover:text-red-300 underline"
              >
                Yes, purge
              </button>
              <button
                onClick={() => setConfirmPurgeAll(false)}
                className="text-xs text-slate-500 hover:text-slate-300"
              >
                Cancel
              </button>
            </div>
          )}
          <button onClick={load} className="btn-ghost" disabled={loading}>
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} onRetry={load} compact />}

      {/* Info banner if empty */}
      {!loading && total === 0 && (
        <div className="flex flex-col items-center justify-center min-h-[300px] gap-4 glass rounded-xl">
          <div className="p-4 rounded-full bg-emerald-400/10 border border-emerald-500/20">
            <AlertTriangle className="w-8 h-8 text-emerald-400" />
          </div>
          <div className="text-center">
            <p className="text-slate-300 font-semibold">DLQ is empty</p>
            <p className="text-sm text-slate-500 mt-1">No failed workflow entries</p>
          </div>
        </div>
      )}

      {/* Table */}
      {(loading || total > 0) && (
        <div className="glass rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left">
                <th className="px-5 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Workflow
                </th>
                <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider hidden md:table-cell">
                  Error
                </th>
                <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Retries
                </th>
                <th className="px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider hidden lg:table-cell">
                  Created
                </th>
                <th className="px-5 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider text-right">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loading && entries.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-16">
                    <FullPageSpinner />
                  </td>
                </tr>
              ) : (
                entries.map((entry) => (
                  <tr
                    key={entry.id}
                    className="border-b border-slate-800/60 last:border-0 table-row-hover group"
                  >
                    <td className="px-5 py-3.5">
                      <p className="text-slate-200 font-medium font-mono text-xs truncate max-w-[160px]">
                        {entry.workflow_id}
                      </p>
                      {entry.step_id && (
                        <p className="text-xs text-slate-600 font-mono mt-0.5 truncate max-w-[160px]">
                          step: {entry.step_id}
                        </p>
                      )}
                    </td>
                    <td className="px-4 py-3.5 hidden md:table-cell">
                      <p className="text-xs text-red-400/80 truncate max-w-[280px]">
                        {entry.error}
                      </p>
                    </td>
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-1.5">
                        <RotateCcw className="w-3.5 h-3.5 text-yellow-500/60" />
                        <span
                          className={`text-sm font-semibold tabular-nums ${
                            entry.retry_count >= 3 ? 'text-red-400' : 'text-yellow-400'
                          }`}
                        >
                          {entry.retry_count}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3.5 hidden lg:table-cell">
                      <span className="text-xs text-slate-500 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(entry.created_at)}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          title="View details"
                          onClick={() => setSelected(entry)}
                          className="btn-ghost p-1.5"
                        >
                          <Info className="w-3.5 h-3.5" />
                        </button>
                        <button
                          title="Retry"
                          onClick={() => handleRetry(entry.id)}
                          disabled={actionLoading === `retry-${entry.id}`}
                          className="btn-ghost p-1.5 text-yellow-400 hover:text-yellow-300 hover:bg-yellow-400/10"
                        >
                          <RotateCcw className="w-3.5 h-3.5" />
                        </button>
                        <button
                          title="Purge"
                          onClick={() => handlePurge(entry.id)}
                          disabled={actionLoading === `purge-${entry.id}`}
                          className="btn-danger p-1.5"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="flex items-center justify-between px-5 py-3 border-t border-slate-800">
              <p className="text-xs text-slate-500">
                Page {page} of {totalPages}
              </p>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="btn-ghost p-1.5 disabled:opacity-40"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="btn-ghost p-1.5 disabled:opacity-40"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Detail Modal */}
      {selected && (
        <DetailModal
          entry={selected}
          onClose={() => setSelected(null)}
          onRetry={() => handleRetry(selected.id)}
          onPurge={() => handlePurge(selected.id)}
        />
      )}
    </div>
  )
}
