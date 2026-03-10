import { CheckCircle2, XCircle, Clock, PlayCircle, Ban, SkipForward } from 'lucide-react'

type Status =
  | 'completed'
  | 'failed'
  | 'running'
  | 'pending'
  | 'cancelled'
  | 'skipped'
  | 'healthy'
  | 'degraded'
  | 'unhealthy'
  | 'open'
  | 'closed'
  | 'half_open'
  | string

interface StatusBadgeProps {
  status: Status
  size?: 'sm' | 'md'
  showIcon?: boolean
}

const CONFIG: Record<string, { color: string; icon?: React.ReactNode; label?: string }> = {
  completed: {
    color: 'text-emerald-400 bg-emerald-400/10 border-emerald-500/20',
    icon: <CheckCircle2 className="w-3 h-3" />,
  },
  failed: {
    color: 'text-red-400 bg-red-400/10 border-red-500/20',
    icon: <XCircle className="w-3 h-3" />,
  },
  running: {
    color: 'text-brand-400 bg-brand-400/10 border-brand-500/20',
    icon: (
      <span className="inline-block w-2 h-2 rounded-full bg-brand-400 animate-pulse" />
    ),
  },
  pending: {
    color: 'text-slate-400 bg-slate-400/10 border-slate-500/20',
    icon: <Clock className="w-3 h-3" />,
  },
  cancelled: {
    color: 'text-orange-400 bg-orange-400/10 border-orange-500/20',
    icon: <Ban className="w-3 h-3" />,
  },
  skipped: {
    color: 'text-slate-500 bg-slate-500/10 border-slate-600/20',
    icon: <SkipForward className="w-3 h-3" />,
  },
  healthy: {
    color: 'text-emerald-400 bg-emerald-400/10 border-emerald-500/20',
    icon: <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />,
  },
  degraded: {
    color: 'text-yellow-400 bg-yellow-400/10 border-yellow-500/20',
    icon: <span className="inline-block w-2 h-2 rounded-full bg-yellow-400" />,
  },
  unhealthy: {
    color: 'text-red-400 bg-red-400/10 border-red-500/20',
    icon: <span className="inline-block w-2 h-2 rounded-full bg-red-400 animate-pulse" />,
  },
  open: {
    color: 'text-red-400 bg-red-400/10 border-red-500/20',
    label: 'Open',
  },
  closed: {
    color: 'text-emerald-400 bg-emerald-400/10 border-emerald-500/20',
    label: 'Closed',
  },
  half_open: {
    color: 'text-yellow-400 bg-yellow-400/10 border-yellow-500/20',
    label: 'Half Open',
  },
  active: {
    color: 'text-brand-400 bg-brand-400/10 border-brand-500/20',
    icon: <PlayCircle className="w-3 h-3" />,
  },
}

export default function StatusBadge({
  status,
  size = 'md',
  showIcon = true,
}: StatusBadgeProps) {
  const cfg = CONFIG[status] ?? {
    color: 'text-slate-400 bg-slate-400/10 border-slate-500/20',
  }

  const label = cfg.label ?? status.replace(/_/g, ' ')
  const sizeClass = size === 'sm' ? 'px-1.5 py-0 text-[10px] gap-1' : 'px-2.5 py-0.5 text-xs gap-1.5'

  return (
    <span
      className={`badge border font-medium capitalize ${cfg.color} ${sizeClass}`}
    >
      {showIcon && cfg.icon}
      {label}
    </span>
  )
}
