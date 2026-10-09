import { useEffect, useRef, useState } from 'react'

import MessageBubble from './MessageBubble.jsx'

const EXAMPLE_QUESTIONS = [
  'What is the expense ratio of HDFC Large Cap Fund?',
  'What is the lock-in period for an ELSS scheme?',
  'How do I download my capital-gains statement?',
]

export default function ChatView({ messages, streaming, loading, onSend }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView()
  }, [messages])

  function submit() {
    const text = input.trim()
    if (!text || streaming || loading) return
    setInput('')
    onSend(text)
  }

  const empty = messages.length === 0

  return (
    <div className="chat-view">
      <div className="message-list">
        {loading && <p className="muted center">Loading conversation…</p>}

        {!loading && empty && !streaming && (
          <div className="empty-state">
            <h1>Welcome</h1>
            <p>
              Ask a question about the knowledge base — every answer is grounded in
              sources you can expand and verify.
            </p>
            <div className="examples">
              {EXAMPLE_QUESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="example-btn"
                  onClick={() => onSend(question)}
                >
                  {question}
                </button>
              ))}
            </div>
            <p className="disclaimer">Facts-only. No investment advice.</p>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)"
          rows={2}
          disabled={streaming || loading}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={streaming || loading || !input.trim()}
        >
          {streaming ? 'Answering…' : 'Send'}
        </button>
        <p className="disclaimer composer-disclaimer">
          Facts-only. No investment advice.
        </p>
      </form>
    </div>
  )
}
