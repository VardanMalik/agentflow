import type { LucideIcon } from 'lucide-react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface StatsCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: LucideIcon
  iconColor?: string
  trend?: number
  trendLabel?: string
  accent?: 'blue' | 'green' | 'red' | 'yellow' | 'violet'
}

const ACCENT_CLASSES: Record<string, string> = {
  blue:   'text-brand-400 bg-brand-400/10',
  green:  'text-emerald-400 bg-emerald-400/10',
  red:    'text-red-400 bg-red-400/10',
  yellow: 'text-yellow-400 bg-yellow-400/10',
  violet: 'text-violet-400 bg-violet-400/10',
}

export default function StatsCard({
  title,
  value,
  subtitle,
  icon: Icon,
  accent = 'blue',
  trend,
  trendLabel,
}: StatsCardProps) {
  const accentClass = ACCENT_CLASSES[accent] ?? ACCENT_CLASSES.blue

  const TrendIcon =
    trend === undefined ? Minus
    : trend > 0 ? TrendingUp
    : TrendingDown

  const trendColor =
    trend === undefined ? 'text-slate-500'
    : trend > 0 ? 'text-emerald-400'
    : 'text-red-400'

  return (
    <div className="stat-card animate-fade-in">
      <div className="flex items-start justify-between mb-4">
        <div className={`p-2.5 rounded-lg ${accentClass}`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend !== undefined && (
          <div className={`flex items-center gap-1 text-xs font-medium ${trendColor}`}>
            <TrendIcon className="w-3.5 h-3.5" />
            {Math.abs(trend)}%
          </div>
        )}
      </div>

      <div>
        <p className="text-2xl font-bold text-slate-100 tabular-nums">{value}</p>
        <p className="text-sm text-slate-400 mt-0.5">{title}</p>
        {(subtitle || trendLabel) && (
          <p className="text-xs text-slate-500 mt-1">{subtitle ?? trendLabel}</p>
        )}
      </div>
    </div>
  )
}
