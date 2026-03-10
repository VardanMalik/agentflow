import { AlertTriangle, RefreshCw } from 'lucide-react'

interface ErrorMessageProps {
  title?: string
  message: string
  onRetry?: () => void
  compact?: boolean
}

export default function ErrorMessage({
  title = 'Something went wrong',
  message,
  onRetry,
  compact = false,
}: ErrorMessageProps) {
  if (compact) {
    return (
      <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/10 border border-red-500/20 rounded-lg px-3 py-2">
        <AlertTriangle className="w-4 h-4 shrink-0" />
        <span>{message}</span>
        {onRetry && (
          <button
            onClick={onRetry}
            className="ml-auto text-red-400 hover:text-red-300 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center gap-4 min-h-[200px] text-center p-8">
      <div className="p-4 rounded-full bg-red-400/10 border border-red-500/20">
        <AlertTriangle className="w-8 h-8 text-red-400" />
      </div>
      <div>
        <p className="text-slate-200 font-semibold">{title}</p>
        <p className="text-sm text-slate-400 mt-1 max-w-sm">{message}</p>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="btn-ghost">
          <RefreshCw className="w-4 h-4" />
          Try again
        </button>
      )}
    </div>
  )
}
