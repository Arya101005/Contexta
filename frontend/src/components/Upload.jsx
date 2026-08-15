import { useRef } from "react";

export default function Upload({ documents, onUpload, onDelete, activeId, onSelect }) {
  const inputRef = useRef();

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    e.target.value = "";
  };

  return (
    <aside className="panel documents">
      <div className="panel-head">
        <h2>Documents</h2>
        <button className="icon-btn" title="Upload PDF" onClick={() => inputRef.current.click()}>
          +
        </button>
        <input ref={inputRef} type="file" accept="application/pdf" hidden onChange={handleFile} />
      </div>

      <div className="doc-list">
        {documents.length === 0 && <p className="muted">No documents yet. Upload a PDF to begin.</p>}
        {documents.map((doc) => (
          <div
            key={doc.id}
            className={`doc-item ${activeId === doc.id ? "active" : ""}`}
            onClick={() => onSelect(doc.id)}
          >
            <span className={`status ${doc.status === "processed" ? "ok" : "pending"}`}>
              {doc.status === "processed" ? "✓" : "…"}
            </span>
            <div className="doc-meta">
              <span className="doc-name">{doc.filename}</span>
              <span className="doc-sub">
                {doc.page_count ? `${doc.page_count} pages · ` : ""}
                {doc.status}
              </span>
            </div>
            <button
              className="del-btn"
              title="Delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(doc.id);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
