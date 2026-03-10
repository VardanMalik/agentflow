import axios from 'axios'
import { useEffect, useRef, useState, useCallback } from 'react'

const BASE_URL = import.meta.env.VITE_API_URL ?? '/api'

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000,
})

apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const message = err.response?.data?.detail ?? err.message ?? 'Request failed'
    return Promise.reject(new Error(message))
  },
)

// ─── Types ───────────────────────────────────────────────────────────────────

export interface Workflow {
  id: string
  name: string
  description?: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  steps: WorkflowStep[]
  created_at: string
  updated_at: string
  duration_ms?: number
  error?: string
}

export interface WorkflowStep {
  id: string
  name: string
  agent_type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  started_at?: string
  completed_at?: string
  duration_ms?: number
  output?: Record<string, unknown>
  error?: string
  tokens_used?: number
}

export interface WorkflowListResponse {
  workflows: Workflow[]
  total: number
  page: number
  page_size: number
}

export interface AgentType {
  type: string
  description: string
  execution_count: number
  success_count: number
  failure_count: number
  avg_duration_ms: number
  total_tokens: number
}

export interface DLQEntry {
  id: string
  workflow_id: string
  step_id?: string
  error: string
  payload: Record<string, unknown>
  created_at: string
  retry_count: number
}

export interface DLQResponse {
  entries: DLQEntry[]
  total: number
}

export interface DashboardStats {
  total_workflows: number
  running_workflows: number
  completed_workflows: number
  failed_workflows: number
  success_rate: number
  avg_duration_ms: number
  active_agents: number
  dlq_size: number
}

export interface HealthStatus {
  status: string
  circuit_breaker: string
  bulkhead: { active: number; max: number }
  dlq_size: number
  uptime_seconds: number
}

export interface RecentActivity {
  id: string
  workflow_id: string
  workflow_name: string
  event: string
  timestamp: string
  status: string
}

export interface ThroughputPoint {
  timestamp: string
  completed: number
  failed: number
  running: number
}

// ─── API Methods ─────────────────────────────────────────────────────────────

export const api = {
  // Workflows
  getWorkflows: (params?: {
    page?: number
    page_size?: number
    status?: string
  }) => apiClient.get<WorkflowListResponse>('/workflows', { params }).then((r) => r.data),

  getWorkflow: (id: string) =>
    apiClient.get<Workflow>(`/workflows/${id}`).then((r) => r.data),

  executeWorkflow: (id: string, input?: Record<string, unknown>) =>
    apiClient.post<{ execution_id: string }>(`/workflows/${id}/execute`, { input }).then((r) => r.data),

  cancelWorkflow: (id: string) =>
    apiClient.post(`/workflows/${id}/cancel`).then((r) => r.data),

  retryWorkflow: (id: string) =>
    apiClient.post(`/workflows/${id}/retry`).then((r) => r.data),

  // Agents
  getAgentTypes: () =>
    apiClient.get<AgentType[]>('/agents/types').then((r) => r.data),

  // DLQ
  getDLQ: (params?: { page?: number; page_size?: number }) =>
    apiClient.get<DLQResponse>('/dlq', { params }).then((r) => r.data),

  retryDLQEntry: (id: string) =>
    apiClient.post(`/dlq/${id}/retry`).then((r) => r.data),

  purgeDLQEntry: (id: string) =>
    apiClient.delete(`/dlq/${id}`).then((r) => r.data),

  purgeAllDLQ: () =>
    apiClient.delete('/dlq').then((r) => r.data),

  // Dashboard
  getStats: () =>
    apiClient.get<DashboardStats>('/dashboard/stats').then((r) => r.data),

  getRecentActivity: (limit = 10) =>
    apiClient.get<RecentActivity[]>('/dashboard/recent', { params: { limit } }).then((r) => r.data),

  getThroughput: (window_minutes = 60) =>
    apiClient.get<ThroughputPoint[]>('/dashboard/throughput', { params: { window_minutes } }).then((r) => r.data),

  getHealth: () =>
    apiClient.get<HealthStatus>('/health').then((r) => r.data),
}

// ─── WebSocket Hook ───────────────────────────────────────────────────────────

export interface WSMessage {
  type: string
  workflow_id?: string
  step_id?: string
  status?: string
  data?: unknown
  timestamp: string
}

interface UseWebSocketReturn {
  messages: WSMessage[]
  connected: boolean
  send: (payload: unknown) => void
  clearMessages: () => void
}

export function useWebSocket(workflowId?: string): UseWebSocketReturn {
  const [messages, setMessages] = useState<WSMessage[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    const wsBase = import.meta.env.VITE_WS_URL ?? `ws://${window.location.host}/ws`
    const url = workflowId ? `${wsBase}?workflow_id=${workflowId}` : wsBase

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (mountedRef.current) setConnected(true)
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const msg: WSMessage = JSON.parse(event.data as string)
          setMessages((prev) => [...prev.slice(-199), msg])
        } catch {
          // ignore parse errors
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setConnected(false)
        reconnectRef.current = setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws.close()
      }
    } catch {
      reconnectRef.current = setTimeout(connect, 5000)
    }
  }, [workflowId])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((payload: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload))
    }
  }, [])

  const clearMessages = useCallback(() => setMessages([]), [])

  return { messages, connected, send, clearMessages }
}
