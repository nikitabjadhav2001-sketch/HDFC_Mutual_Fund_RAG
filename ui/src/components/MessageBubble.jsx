import { useState } from 'react'

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'
  const sources = message.sources || []

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-role">{isUser ? 'You' : 'Assistant'}</div>
      <div className="message-content">
        {message.pending && !message.content ? (
          <span className="thinking">
            Thinking<span className="dots">…</span>
          </span>
        ) : (
          message.content
        )}
      </div>
      {!isUser && sources.length > 0 && (
        <div className="sources">
          <button
            type="button"
            className="sources-toggle"
            onClick={() => setShowSources((visible) => !visible)}
            aria-expanded={showSources}
          >
            {showSources ? 'Hide sources' : `Sources (${sources.length})`}
          </button>
          {showSources && (
            <ul className="sources-list">
              {sources.map((source, index) => (
                <li key={source.chunk_id || index} className="source">
                  <div className="source-header">
                    <span className="source-title">{source.doc_title || 'Document'}</span>
                    {source.page != null && <span className="source-page">p. {source.page}</span>}
                  </div>
                  {source.snippet && <p className="source-snippet">{source.snippet}…</p>}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
