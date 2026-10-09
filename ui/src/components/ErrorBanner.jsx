export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="error-banner" role="alert">
      <span>{message}</span>
      <button type="button" className="icon-btn" aria-label="Dismiss" onClick={onDismiss}>
        ×
      </button>
    </div>
  )
}
