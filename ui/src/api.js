// Base URL for the API. Empty (default) = same-origin: Vite dev proxy in
// development, nginx proxy in docker. Set VITE_API_BASE_URL to override.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')
const APP_KEY = import.meta.env.VITE_API_KEY
const ADMIN_KEY = import.meta.env.VITE_ADMIN_KEY || APP_KEY

function authHeaders() {
  return APP_KEY ? { Authorization: `Bearer ${APP_KEY}` } : {}
}

function adminHeaders() {
  return ADMIN_KEY ? { Authorization: `Bearer ${ADMIN_KEY}` } : {}
}

async function request(path, options = {}) {
  const headers = { ...authHeaders(), ...(options.headers || {}) }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch {
      // non-JSON error body
    }
    throw new Error(`${response.status}: ${detail}`)
  }
  if (response.status === 204) return null
  return response.json()
}

function jsonOptions(options = {}) {
  return {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  }
}

export function listConversations() {
  return request('/api/v1/conversations')
}

export function getConversation(id) {
  return request(`/api/v1/conversations/${id}`)
}

export function deleteConversation(id) {
  return request(`/api/v1/conversations/${id}`, { method: 'DELETE' })
}

export async function streamChat({ conversationId, message, history }, onEvent) {
  const response = await fetch(
    `${API_BASE}/api/v1/chat`,
    jsonOptions({
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        conversation_id: conversationId || null,
        message,
        history: history || [],
      }),
    }),
  )
  if (!response.ok || !response.body) {
    throw new Error(`chat request failed (${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data: ')) continue
        try {
          onEvent(JSON.parse(line.slice(6)))
        } catch {
          // ignore malformed frames
        }
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}

export function listDocuments() {
  return request('/api/v1/admin/documents', { headers: adminHeaders() })
}

export function deleteDocument(id) {
  return request(`/api/v1/admin/documents/${id}`, { method: 'DELETE', headers: adminHeaders() })
}

export async function uploadDocument(file) {
  const form = new FormData()
  form.append('file', file)
  const body = await request('/api/v1/admin/documents', {
    method: 'POST',
    body: form,
    headers: adminHeaders(),
  })
  return body.document
}
