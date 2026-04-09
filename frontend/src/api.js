const API_BASE = '/api';

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload-documents`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function processInput(inputText) {
  const res = await fetch(`${API_BASE}/process-input`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input_text: inputText }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Processing failed');
  }
  return res.json();
}

export async function getResults(sessionId) {
  const res = await fetch(`${API_BASE}/get-results/${sessionId}`);
  if (!res.ok) throw new Error('Failed to fetch results');
  return res.json();
}

export async function submitFeedback(claimId, sessionId, decision) {
  const res = await fetch(`${API_BASE}/submit-feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim_id: claimId, session_id: sessionId, decision }),
  });
  if (!res.ok) throw new Error('Failed to submit feedback');
  return res.json();
}

export async function editClaim(claimId, sessionId, newClaimText) {
  const res = await fetch(`${API_BASE}/edit-claim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim_id: claimId, session_id: sessionId, new_claim_text: newClaimText }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Edit failed');
  }
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export async function evaluateReasoning(studentAnswer, problem = '', confidence = 50) {
  const res = await fetch(`${API_BASE}/evaluate-reasoning`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      student_answer: studentAnswer,
      problem,
      confidence,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Stress test failed');
  }
  return res.json();
}

export async function simulateThinking(problem, studentAnswer = '') {
  const res = await fetch(`${API_BASE}/simulate-thinking`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      problem,
      student_answer: studentAnswer,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Thinking simulation failed');
  }
  return res.json();
}
