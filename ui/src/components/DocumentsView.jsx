import { useCallback, useEffect, useState } from 'react'

import { deleteDocument, listDocuments, uploadDocument } from '../api.js'

const ACTIVE_STATUSES = new Set(['queued', 'processing'])

export default function DocumentsView({ onError }) {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments())
    } catch (error) {
      onError(error.message)
    } finally {
      setLoading(false)
    }
  }, [onError])

  useEffect(() => {
    refresh()
  }, [refresh])

  const polling = documents.some((doc) => ACTIVE_STATUSES.has(doc.status))
  useEffect(() => {
    if (!polling) return undefined
    const timer = setInterval(refresh, 2000)
    return () => clearInterval(timer)
  }, [polling, refresh])

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUploading(true)
    try {
      await uploadDocument(file)
      await refresh()
    } catch (error) {
      onError(error.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(document) {
    if (!window.confirm(`Delete "${document.title}"?`)) return
    try {
      await deleteDocument(document.id)
      await refresh()
    } catch (error) {
      onError(error.message)
    }
  }

  return (
    <div className="documents-view">
      <div className="documents-header">
        <h2>Knowledge base</h2>
        <label className={`btn btn-primary${uploading ? ' disabled' : ''}`}>
          {uploading ? 'Uploading…' : 'Upload document'}
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md,.markdown"
            onChange={handleUpload}
            disabled={uploading}
            hidden
          />
        </label>
      </div>

      {loading && <p className="muted">Loading documents…</p>}

      {!loading && documents.length === 0 && (
        <p className="muted">
          No documents yet. Upload a PDF, DOCX, TXT or Markdown file to grow the
          knowledge base.
        </p>
      )}

      {documents.length > 0 && (
        <table className="doc-table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Format</th>
              <th>Status</th>
              <th>Chunks</th>
              <th>Uploaded</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {documents.map((document) => (
              <tr key={document.id}>
                <td>
                  <span className="doc-title">{document.title}</span>
                  {document.error && <div className="doc-error">{document.error}</div>}
                </td>
                <td>
                  <span className="badge">{document.format}</span>
                </td>
                <td>
                  <span className={`badge badge-${document.status}`}>{document.status}</span>
                </td>
                <td>{document.chunk_count}</td>
                <td className="muted">
                  {document.uploaded_at
                    ? new Date(document.uploaded_at).toLocaleString()
                    : '—'}
                </td>
                <td>
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label={`Delete ${document.title}`}
                    onClick={() => handleDelete(document)}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
