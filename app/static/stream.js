// Kept dependency-free so the same streaming protocol can be tested with node --test.
async function streamRequest(path, body, onEvent, signal) {
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body), signal
  });
  if (!response.ok) {
    let data;
    try { data = await response.json(); } catch (_) {}
    throw new Error(data?.detail || `${response.status} ${response.statusText}`);
  }
  if (!response.body) throw new Error('Streaming response has no body.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', completed = false;
  const emit = line => {
    if (!line.trim()) return;
    if (signal?.aborted) throw new DOMException('Generation stopped.', 'AbortError');
    let event;
    try { event = JSON.parse(line); } catch (_) { throw new Error('Malformed streaming response.'); }
    if (!event || typeof event !== 'object' || typeof event.type !== 'string') {
      throw new Error('Malformed streaming response.');
    }
    if (event.type === 'error') {
      onEvent(event);
      const error = new Error(event.message || 'Generation failed.');
      error.reason = event.reason;
      throw error;
    }
    if (event.type === 'done') completed = true;
    onEvent(event);
    if (!completed && signal?.aborted) throw new DOMException('Generation stopped.', 'AbortError');
  };
  const consume = () => {
    let newline;
    while (!completed && (newline = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      emit(line);
    }
  };
  try {
    while (true) {
      if (signal?.aborted) throw new DOMException('Generation stopped.', 'AbortError');
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      consume();
      if (completed) return;
    }
    buffer += decoder.decode();
    consume();
    if (!completed) emit(buffer); // The final event need not end in a newline.
    if (!completed) throw new Error('Stream ended before generation completed.');
  } finally {
    try { await reader.cancel(); } catch (_) {}
    reader.releaseLock();
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = {streamRequest};
