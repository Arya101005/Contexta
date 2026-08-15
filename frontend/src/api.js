const BASE = "/api";

export function uploadDocument(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/documents/upload`, true);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded / e.total);
      }
    });
    xhr.onreadystatechange = () => {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error("Invalid response"));
          }
        } else {
          try {
            reject(new Error((JSON.parse(xhr.responseText)).detail || "Upload failed"));
          } catch {
            reject(new Error("Upload failed"));
          }
        }
      }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.send(form);
  });
}

export async function getDocuments() {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error("Failed to load documents");
  return res.json();
}

export async function deleteDocument(id) {
  const res = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete document");
  return res.json();
}

export async function askQuestion(question, { documentId = null, sessionId = null } = {}) {
  const params = new URLSearchParams({ question });
  if (documentId) params.set("document_id", documentId);
  if (sessionId) params.set("session_id", sessionId);
  const res = await fetch(`${BASE}/chat?${params.toString()}`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).detail || "Query failed");
  return res.json();
}
