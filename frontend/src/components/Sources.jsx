import { useState } from "react";

export default function Sources({ sources, documents }) {
  const [open, setOpen] = useState(null);

  const nameOf = (id) => documents.find((d) => d.doc_id && String(d.doc_id) === String(id))?.filename || "Source";

  if (!sources || sources.length === 0) {
    return (
      <aside className="panel sources">
        <div className="panel-head">
          <h2>Sources</h2>
        </div>
        <p className="muted">Sources cited in the answer will appear here.</p>
      </aside>
    );
  }

  return (
    <aside className="panel sources">
      <div className="panel-head">
        <h2>Sources</h2>
        <span className="count">{sources.length}</span>
      </div>

      <div className="source-list">
        {sources.map((s, i) => (
          <div key={i} className="source-card">
            <button className="source-btn" onClick={() => setOpen(open === i ? null : i)}>
              <span className="src-name">{nameOf(s.document_id)}</span>
              <span className="src-page">Page {s.page ?? "—"}</span>
            </button>
            {open === i && (
              <div className="source-detail">
                <div className="src-row">
                  <span>Section</span>
                  <strong>{s.section || "General"}</strong>
                </div>
                <div className="src-row">
                  <span>Page</span>
                  <strong>{s.page ?? "—"}</strong>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}
