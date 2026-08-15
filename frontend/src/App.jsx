import { useEffect, useState } from "react";
import { askQuestion, deleteDocument, getDocuments, uploadDocument } from "./api";
import Upload from "./components/Upload";
import Chat from "./components/Chat";
import Sources from "./components/Sources";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [error, setError] = useState("");

  const refresh = () => getDocuments().then(setDocuments).catch(() => {});

  useEffect(() => {
    refresh();
  }, []);

  const handleUpload = async (file) => {
    setError("");
    setUploading(true);
    setUploadProgress(0);
    try {
      const doc = await uploadDocument(file, (p) => setUploadProgress(p));
      await refresh();
      setActiveId(doc.document_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  const handleDelete = async (id) => {
    await deleteDocument(id);
    if (activeId === id) setActiveId(null);
    await refresh();
  };

  const handleAsk = async (question) => {
    if (documents.length === 0) {
      setError("Upload at least one document first.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await askQuestion(question, { documentId: activeId, sessionId });
      setMessages((prev) => [
        ...prev,
        { id: res.message_id, question: res.question, answer: res.answer },
      ]);
      setSources(res.sources || []);
      setSessionId(res.session_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◈</span>
          <span>Contexta</span>
        </div>
        <span className="tagline">Ask your documents</span>
        {uploading && (
          <div className="upload-progress">
            <div className="upload-progress-bar" style={{ width: `${(uploadProgress ?? 0) * 100}%` }} />
            <span className="upload-progress-text">{Math.round((uploadProgress ?? 0) * 100)}%</span>
          </div>
        )}
      </header>

      {error && <div className="error-bar">{error}</div>}

      <main className="layout">
        <Upload
          documents={documents}
          activeId={activeId}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onSelect={setActiveId}
        />
        <Chat messages={messages} loading={loading} onAsk={handleAsk} />
        <Sources sources={sources} documents={documents} />
      </main>
    </div>
  );
}
