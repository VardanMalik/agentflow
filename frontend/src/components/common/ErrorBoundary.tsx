import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
  fallbackLabel?: string
}

interface State {
  hasError: boolean
  message: string | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message ?? 'Unexpected error' }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught an error', error, info?.componentStack)
  }

  reset = () => {
    this.setState({ hasError: false, message: null })
  }

  render() {
    if (!this.state.hasError) return this.props.children

    return (
      <div className="glass rounded-xl p-8 m-5 text-center">
        <AlertTriangle className="w-10 h-10 text-red-400 mx-auto mb-3" />
        <h2 className="text-lg font-semibold text-slate-100 mb-1">
          {this.props.fallbackLabel ?? 'Something went wrong'}
        </h2>
        <p className="text-sm text-slate-400 mb-4 break-words">
          {this.state.message}
        </p>
        <button onClick={this.reset} className="btn-ghost inline-flex">
          <RefreshCw className="w-4 h-4" />
          Try again
        </button>
      </div>
    )
  }
}
