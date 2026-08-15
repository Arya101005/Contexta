import { useState } from "react";

export default function Chat({ messages, loading, onAsk }) {
  const [text, setText] = useState("");

  const submit = (e) => {
    e.preventDefault();
    const q = text.trim();
    if (!q || loading) return;
    onAsk(q);
    setText("");
  };

  return (
    <section className="panel chat">
      <div className="chat-head">
        <h2>Ask your documents</h2>
        <p className="muted">Get grounded answers with citations from your files.</p>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="big-q">“</div>
            <p>Try: <em>“What was the revenue growth?”</em></p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className="msg">
            <div className="bubble question">{m.question}</div>
            <div className="bubble answer">{m.answer}</div>
          </div>
        ))}
        {loading && (
          <div className="msg">
            <div className="bubble answer loading">
              <span className="thinking-dots">
                <span>.</span><span>.</span><span>.</span>
              </span>
              <span className="thinking-text">Generating answer</span>
            </div>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask a question about your documents…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !text.trim()}>
          Ask
        </button>
      </form>
    </section>
  );
}
