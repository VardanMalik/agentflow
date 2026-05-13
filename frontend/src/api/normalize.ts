export function toArray<T>(data: unknown): T[] {
  if (Array.isArray(data)) return data as T[]
  if (data && typeof data === 'object') {
    const obj = data as Record<string, unknown>
    if (Array.isArray(obj.items)) return obj.items as T[]
    if (Array.isArray(obj.workflows)) return obj.workflows as T[]
    if (Array.isArray(obj.entries)) return obj.entries as T[]
    if (Array.isArray(obj.results)) return obj.results as T[]
    if (Array.isArray(obj.data)) return obj.data as T[]
  }
  return []
}

export function toTotal(data: unknown, fallback: number): number {
  if (data && typeof data === 'object') {
    const t = (data as Record<string, unknown>).total
    if (typeof t === 'number') return t
  }
  return fallback
}
