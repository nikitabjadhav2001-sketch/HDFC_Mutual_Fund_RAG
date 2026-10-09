export default function Sidebar({
  conversations,
  activeId,
  loading,
  onSelect,
  onNew,
  onDelete,
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <button type="button" className="btn btn-primary" onClick={onNew}>
          + New chat
        </button>
      </div>
      <nav className="conversation-list" aria-label="Conversations">
        {loading && <p className="muted">Loading…</p>}
        {!loading && conversations.length === 0 && (
          <p className="muted">No conversations yet.</p>
        )}
        {conversations.map((conversation) => (
          <div
            key={conversation.id}
            className={`conversation-row${conversation.id === activeId ? ' active' : ''}`}
          >
            <button
              type="button"
              className="conversation-title"
              onClick={() => onSelect(conversation.id)}
              title={conversation.title || 'Untitled chat'}
            >
              {conversation.title || 'Untitled chat'}
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label="Delete conversation"
              onClick={() => onDelete(conversation.id)}
            >
              ×
            </button>
          </div>
        ))}
      </nav>
    </aside>
  )
}
