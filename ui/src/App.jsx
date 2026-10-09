import { useCallback, useEffect, useState } from 'react'

import { deleteConversation, getConversation, listConversations, streamChat } from './api.js'
import ChatView from './components/ChatView.jsx'
import DocumentsView from './components/DocumentsView.jsx'
import ErrorBanner from './components/ErrorBanner.jsx'
import Sidebar from './components/Sidebar.jsx'

function uid() {
  return crypto.randomUUID()
}

export default function App() {
  const [view, setView] = useState('chat')
  const [conversations, setConversations] = useState([])
  const [conversationsLoading, setConversationsLoading] = useState(true)
  const [activeId, setActiveId] = useState(null)
  const [messages, setMessages] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [conversationLoading, setConversationLoading] = useState(false)
  const [error, setError] = useState(null)

  const showError = useCallback((message) => setError(message), [])

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await listConversations())
    } catch (err) {
      setError(err.message)
    } finally {
      setConversationsLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshConversations()
  }, [refreshConversations])

  const openConversation = useCallback(async (id) => {
    setActiveId(id)
    setMessages([])
    if (!id) return
    setConversationLoading(true)
    try {
      const conversation = await getConversation(id)
      setMessages(
        conversation.messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          sources: message.citations || [],
        })),
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setConversationLoading(false)
    }
  }, [])

  const newChat = useCallback(() => {
    setActiveId(null)
    setMessages([])
    setError(null)
    setView('chat')
  }, [])

  const removeConversation = useCallback(
    async (id) => {
      if (!window.confirm('Delete this conversation?')) return
      try {
        await deleteConversation(id)
        if (id === activeId) {
          setActiveId(null)
          setMessages([])
        }
        await refreshConversations()
      } catch (err) {
        setError(err.message)
      }
    },
    [activeId, refreshConversations],
  )

  const send = useCallback(
    async (text) => {
      const trimmed = text.trim()
      if (!trimmed || streaming) return
      setError(null)
      setView('chat')

      const history = messages.map((message) => ({
        role: message.role,
        content: message.content,
      }))
      const assistantId = uid()
      setMessages((previous) => [
        ...previous,
        { id: uid(), role: 'user', content: trimmed },
        { id: assistantId, role: 'assistant', content: '', sources: [], pending: true },
      ])
      setStreaming(true)

      const settleAssistant = (patch) =>
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId ? { ...message, pending: false, ...patch } : message,
          ),
        )

      try {
        await streamChat(
          { conversationId: activeId, message: trimmed, history },
          (event) => {
            if (event.type === 'token') {
              setMessages((previous) =>
                previous.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content + event.content, pending: false }
                    : message,
                ),
              )
            } else if (event.type === 'sources') {
              setMessages((previous) =>
                previous.map((message) =>
                  message.id === assistantId
                    ? { ...message, sources: event.sources || [] }
                    : message,
                ),
              )
            } else if (event.type === 'done') {
              settleAssistant({})
              if (event.conversation_id) setActiveId(event.conversation_id)
              refreshConversations()
            } else if (event.type === 'error') {
              settleAssistant({})
              if (event.conversation_id) setActiveId(event.conversation_id)
              setError(event.message || 'Chat request failed')
            }
          },
        )
      } catch (err) {
        settleAssistant({})
        setError(err.message)
      } finally {
        setStreaming(false)
      }
    },
    [streaming, messages, activeId, refreshConversations],
  )

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">RAG Chatbot</span>
        <nav className="tabs" aria-label="Views">
          <button
            type="button"
            className={`tab${view === 'chat' ? ' active' : ''}`}
            onClick={() => setView('chat')}
          >
            Chat
          </button>
          <button
            type="button"
            className={`tab${view === 'documents' ? ' active' : ''}`}
            onClick={() => setView('documents')}
          >
            Documents
          </button>
        </nav>
      </header>

      <div className="layout">
        {view === 'chat' && (
          <Sidebar
            conversations={conversations}
            activeId={activeId}
            loading={conversationsLoading}
            onSelect={openConversation}
            onNew={newChat}
            onDelete={removeConversation}
          />
        )}

        <main className="main">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
          {view === 'chat' ? (
            <ChatView
              messages={messages}
              streaming={streaming}
              loading={conversationLoading}
              onSend={send}
            />
          ) : (
            <DocumentsView onError={showError} />
          )}
        </main>
      </div>
    </div>
  )
}
