const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  notebooks: [],
  current: null,
  detail: null,
  messages: [],
  generating: false,
  run: null,
  viewEpoch: 0,
  selecting: null,
  editing: null,
  poller: null,
  sourceQuery: '',
  numCtx: null
};

// Emojis for notebook thumbnails
const EMOJIS = ['🛣️', '🇪🇺', '🔬', '🚢', '📈', '✉️', '📄', '🧠', '📊', '🌐', '📚', '💡'];

function getEmojiForNotebook(id) {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash << 5) - hash + id.charCodeAt(i);
  return EMOJIS[Math.abs(hash) % EMOJIS.length];
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = body.detail || message; } catch (_) {}
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(message) {
  const el = document.createElement('div'); el.className = 'toast'; el.textContent = message;
  $('#toasts').append(el); setTimeout(() => el.remove(), 4200);
}

function selectedSources() {
  return (state.detail?.sources || []).filter(s => s.enabled && s.status === 'ready').map(s => s.id);
}

function formatUsageSplit(gpuPercent, cpuPercent) {
  if (gpuPercent == null || cpuPercent == null) return '';
  if (gpuPercent >= 100) return '100% GPU';
  if (gpuPercent <= 0) return '100% CPU';
  return `${cpuPercent}% CPU / ${gpuPercent}% GPU`;
}

async function loadHealth() {
  try {
    const health = await api('/api/health');
    state.numCtx = health.num_ctx || null;
    const el = $('#modelStatus');
    const label = el.querySelector('.status-label');
    if (!health.ollama) {
      el.className = 'status bad'; el.title = 'Ollama offline';
      if (label) label.textContent = 'Ollama offline';
    } else if (!health.generation_ready) {
      el.className = 'status bad'; el.title = 'Generation model not pulled in Ollama';
      if (label) label.textContent = 'Model missing';
    } else if (!health.model_loaded) {
      el.className = 'status bad'; el.title = `${health.generation_model} not loaded in VRAM`;
      if (label) label.textContent = 'Not loaded';
    } else {
      el.className = 'status good';
      const usage = formatUsageSplit(health.gpu_percent, health.cpu_percent);
      el.title = `${health.generation_model} loaded${usage ? ' · ' + usage : ''}`;
      if (label) label.textContent = usage || health.generation_model;
    }
  } catch (error) { $('#modelStatus').className = 'status bad'; }
}

async function loadSettings() {
  try {
    const settings = await api('/api/settings');
    state.numCtx = settings.num_ctx || state.numCtx;
    const modelSelect = $('#modelSelect'), ctxSelect = $('#ctxSelect');
    if (modelSelect) {
      const models = settings.available_models?.length ? settings.available_models : [settings.generation_model];
      modelSelect.innerHTML = models.map(m => `<option value="${escapeHtml(m)}" ${m === settings.generation_model ? 'selected' : ''}>${escapeHtml(m)}</option>`).join('');
    }
    if (ctxSelect) {
      ctxSelect.innerHTML = (settings.context_options || []).map(n =>
        `<option value="${n}" ${n === settings.num_ctx ? 'selected' : ''}>${formatTokens(n)}</option>`).join('');
    }
  } catch (error) { /* Settings are optional; ignore if Ollama is unreachable. */ }
}

async function changeSettings(patch) {
  try {
    const updated = await api('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(patch)});
    state.numCtx = updated.num_ctx;
    const meter = $('#contextMeter'); if (meter) meter.classList.add('hidden');
    await loadHealth();
    toast('Model settings updated.');
  } catch (error) { toast(`Could not update settings: ${error.message}`); loadSettings(); }
}

async function loadNotebooks(selectId) {
  state.notebooks = await api('/api/notebooks');
  renderNotebookGrid();

  if (selectId) {
    await selectNotebook(selectId);
  } else if (state.current && state.notebooks.some(n => n.id === state.current)) {
    await selectNotebook(state.current);
  } else {
    showLandingView();
  }
}

function renderNotebookGrid() {
  const container = $('#notebookGrid');
  const createCardHtml = `
    <button class="create-card" id="cardCreateBtn">
      <div class="plus-circle">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      </div>
      <span>Create new notebook</span>
    </button>`;

  const cardsHtml = state.notebooks.map(n => {
    const emoji = getEmojiForNotebook(n.id);
    const dateStr = new Date(n.updated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    const count = n.source_count || 0;
    return `
      <div class="notebook-card" data-notebook="${n.id}">
        <button class="card-menu-btn" data-notebook-menu="${n.id}" title="Options">•••</button>
        <div>
          <div class="card-emoji">${emoji}</div>
          <div class="card-title">${escapeHtml(n.title)}</div>
        </div>
        <div class="card-subtitle">${dateStr} · ${count} source${count === 1 ? '' : 's'}</div>
      </div>`;
  }).join('');

  container.innerHTML = createCardHtml + cardsHtml;

  const createBtn = $('#cardCreateBtn');
  if (createBtn) createBtn.onclick = createNotebook;
}

function showLandingView() {
  state.viewEpoch++;
  state.current = null;
  state.detail = null;
  $('#landingView').classList.remove('hidden');
  $('#workspace').classList.add('hidden');
  $('#notebookNavTitle').classList.add('hidden');
  $('#landingNavTabs').classList.remove('hidden');
  renderGenerationStatus();
}

async function selectNotebook(id) {
  const viewEpoch = ++state.viewEpoch;
  state.current = id;
  const selection = {id, viewEpoch, promise: Promise.all([api(`/api/notebooks/${id}`), api(`/api/notebooks/${id}/messages`)])};
  state.selecting = selection;
  try {
    const [detail, messages] = await selection.promise;
    if (state.current !== id || state.viewEpoch !== viewEpoch) return;
    state.detail = detail; state.messages = messages;

    $('#landingView').classList.add('hidden');
    $('#workspace').classList.remove('hidden');
    $('#notebookNavTitle').classList.remove('hidden');
    $('#landingNavTabs').classList.add('hidden');

    const emoji = getEmojiForNotebook(id);
    $('#currentTitle').textContent = state.detail.title;
    $('#bannerTitle').textContent = state.detail.title;
    $('#bannerEmoji').textContent = emoji;

    renderSources();
    renderMessages();
    renderSaved();
    updateSuggestions();
    schedulePolling();
    renderGenerationStatus();
  } finally {
    if (state.selecting === selection) state.selecting = null;
  }
}

function renderSources() {
  let sources = state.detail?.sources || [];
  const totalCount = sources.length;
  const readyCount = sources.filter(s => s.status === 'ready' && s.enabled).length;

  $('#sourceCountLabel').textContent = `${readyCount}/${totalCount} active`;
  $('#bannerMeta').textContent = `${totalCount} source${totalCount === 1 ? '' : 's'} · ${new Date(state.detail.updated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
  $('#composerSourceCount').textContent = `${readyCount} source${readyCount === 1 ? '' : 's'}`;

  if (state.sourceQuery) {
    const q = state.sourceQuery.toLowerCase();
    sources = sources.filter(s => s.name.toLowerCase().includes(q));
  }

  $('#sourceList').innerHTML = sources.map(s => `
    <div class="source-item-row source-${s.status}" data-source-id="${s.id}">
      <svg class="pdf-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      <span class="source-title-wrap">
        <button class="source-title" data-source-details="${s.id}" title="${escapeHtml(s.name)}">${escapeHtml(s.name)}</button>
        <small class="source-state ${s.status}" title="${escapeHtml(s.error || s.status)}">${s.status === 'ready' ? (s.page_count ? `${s.page_count} pages` : `${Math.round(s.char_count/1000)}k chars`) : s.status === 'error' ? 'Failed — retry available' : s.status}</small>
      </span>
      ${s.status === 'error' ? `<button class="source-retry-btn" data-source-retry="${s.id}" title="Retry processing">↻</button>` : ''}
      <button class="source-action-btn" data-source-delete="${s.id}" title="Delete source">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </button>
      <input type="checkbox" class="source-checkbox" data-toggle-source="${s.id}" ${s.enabled ? 'checked' : ''} ${s.status !== 'ready' ? 'disabled' : ''} title="Toggle source active in context">
    </div>`).join('') || '<div class="studio-empty-prompt" style="padding:20px 0;"><p class="empty-desc">No matching sources.</p></div>';
}

function schedulePolling() {
  clearTimeout(state.poller);
  if (!state.detail?.sources.some(s => ['queued','processing'].includes(s.status))) return;
  const notebookId = state.current, viewEpoch = state.viewEpoch;
  state.poller = setTimeout(async () => {
    if (state.current !== notebookId || state.viewEpoch !== viewEpoch) return;
    try {
      const detail = await api(`/api/notebooks/${notebookId}`);
      if (state.current !== notebookId || state.viewEpoch !== viewEpoch) return;
      state.detail = detail; renderSources(); renderSaved(); schedulePolling();
    }
    catch (_) {}
  }, 1800);
}

function renderMessages() {
  const area = $('#messages');
  $('#notebookBanner').classList.toggle('hidden', state.messages.length > 0);
  area.querySelectorAll('.message').forEach(el => el.remove());
  state.messages.forEach(message => appendMessage(message.role, message.content, message.citations || [], false));
  area.scrollTop = area.scrollHeight;
}

function appendMessage(role, content, citations = [], streaming = false) {
  const el = document.createElement('article'); el.className = `message ${role}`;
  const avatarText = role === 'assistant' ? 'G' : 'You';
  const actionsHtml = role === 'assistant' && !streaming ? `
    <div class="message-actions">
      <button class="text-button copy-answer">Copy</button>
      <button class="text-button save-answer">Save as note</button>
    </div>` : '';

  el.innerHTML = `
    <div class="avatar">${avatarText}</div>
    <div>
      <div class="message-body ${streaming ? 'typing' : ''}">${markdown(content, citations)}</div>
      ${actionsHtml}
    </div>`;
  el._content = content; el._citations = citations;
  $('#messages').append(el); $('#messages').scrollTop = $('#messages').scrollHeight;
  return el;
}

function updateMessage(el, content, citations, streaming, status = '') {
  el._content = content; el._citations = citations;
  const body = el.querySelector('.message-body');
  body.innerHTML = (status ? `<span class="assistant-progress" role="status" aria-live="polite">${escapeHtml(status)}</span>` : '') + markdown(content, citations);
  body.classList.toggle('typing', streaming);
  $('#messages').scrollTop = $('#messages').scrollHeight;
}

function updateSuggestions() {
  const sources = state.detail?.sources.filter(s => s.status === 'ready') || [];
  const options = sources.length > 1
    ? ['What are the main themes across these sources?', 'Where do the sources agree or disagree?', 'What key evidence is provided?']
    : ['Summarize the core ideas in this source.', 'What key evidence supports the main argument?', 'Extract key definitions and takeaways.'];
  $('#suggestions').innerHTML = options.map(x => `<button class="suggestion-chip">${escapeHtml(x)}</button>`).join('');
}

function formatTokens(n) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n);
}

function updateContextMeter(promptTokens, evalTokens) {
  const el = $('#contextMeter');
  if (!el || !state.numCtx) return;
  const used = promptTokens + evalTokens;
  const pct = Math.min(100, Math.round((used / state.numCtx) * 100));
  el.classList.remove('hidden');
  el.classList.toggle('warn', pct >= 75 && pct < 90);
  el.classList.toggle('danger', pct >= 90);
  el.textContent = `Request ${pct}% · ${formatTokens(used)}/${formatTokens(state.numCtx)}`;
  el.title = `${used.toLocaleString()} tokens in the most recent model request of ${state.numCtx.toLocaleString()} configured context tokens. This is not cumulative notebook or chat memory.`;
}

function renderGenerationStatus() {
  const run = state.run, same = run && state.current === run.notebookId;
  if (run?.kind === 'chat' && same && !run.hasOutput && run.answerEl) {
    updateMessage(run.answerEl, '', run.citations || [], true, run.status);
  }
  $('#studioStatus').textContent = run?.kind === 'studio' ? (same ? run.status : `Studio running in ${run.title}…`) : '';
  $('#sendButton').classList.toggle('hidden', run?.kind === 'chat');
  $('#stopButton').classList.toggle('hidden', run?.kind !== 'chat');
  $('#studioStopButton').classList.toggle('hidden', run?.kind !== 'studio');
  const globalStop = $('#globalStopButton');
  globalStop.classList.toggle('hidden', !run || state.current !== null);
  globalStop.setAttribute('aria-label', run ? `Stop generation in ${run.title}` : 'Stop generation');
}

function responseMetrics(metrics, wallMs, firstTokenMs) {
  const parts = [];
  if (Number.isInteger(metrics?.eval_count) && metrics.eval_count >= 0) {
    parts.push(`${metrics.eval_count} output tokens`);
    if (Number.isInteger(metrics.eval_duration) && metrics.eval_duration > 0) {
      parts.push(`${(metrics.eval_count / (metrics.eval_duration / 1e9)).toFixed(1)} tokens/s`);
    }
  }
  if (Number.isInteger(metrics?.total_duration) && metrics.total_duration >= 0) {
    parts.push(`Model ${(metrics.total_duration / 1e9).toFixed(1)}s`);
  }
  if (firstTokenMs != null) parts.push(`Before first token ${(firstTokenMs / 1000).toFixed(1)}s`);
  parts.push(`Wall ${(wallMs / 1000).toFixed(1)}s`);
  return parts.join(' · ');
}

async function sendQuestion(question) {
  if (!state.current || state.generating || !question.trim()) return;
  if (state.detail?.id !== state.current) return toast('Notebook is still loading.');
  const notebookId = state.current, viewEpoch = state.viewEpoch, title = state.detail.title;
  state.generating = true; $('#notebookBanner').classList.add('hidden'); $('#question').value = ''; resizeComposer();
  appendMessage('user', question.trim());
  const answerEl = appendMessage('assistant', '', [], true); let answer = '', citations = [];
  const rejectedDrafts = [];
  const controller = new AbortController();
  const run = {notebookId, title, kind:'chat', status:'Finding evidence and generating an answer…', controller,
               answerEl, citations, hasOutput:false, startedAt:Date.now(), firstTokenAt:null, metrics:null};
  state.run = run; renderGenerationStatus();
  const originalView = () => state.current === notebookId && state.viewEpoch === viewEpoch;
  try {
    await streamRequest(`/api/notebooks/${notebookId}/chat`, {question:question.trim(), source_ids:selectedSources()}, event => {
      if (event.type === 'citations') { citations = event.citations; run.citations = citations; }
      if (event.type === 'retry') {
        if (answer.trim()) rejectedDrafts.push(answer);
        answer = '';
        run.hasOutput = false;
        run.status = event.message || 'Retrying with source citations…';
        renderGenerationStatus();
      }
      if (event.type === 'delta') {
        if (!run.hasOutput) run.firstTokenAt = Date.now();
        run.hasOutput = true; answer += event.text;
        if (originalView()) updateMessage(answerEl, answer, citations, true);
      }
      if (event.type === 'done' && event.metrics) {
        run.metrics = event.metrics;
        const {prompt_eval_count, eval_count} = event.metrics;
        if (event.num_ctx) state.numCtx = event.num_ctx;
        if (Number.isInteger(prompt_eval_count) && Number.isInteger(eval_count)) {
          updateContextMeter(prompt_eval_count, eval_count);
        }
      }
      if (event.type === 'status' && typeof event.message === 'string') { run.status = event.message; renderGenerationStatus(); }
    }, controller.signal);
    if (originalView()) {
      updateMessage(answerEl, answer, citations, false);
      const stats = document.createElement('div'); stats.className = 'answer-metrics';
      stats.textContent = responseMetrics(run.metrics, Date.now() - run.startedAt,
                                          run.firstTokenAt == null ? null : run.firstTokenAt - run.startedAt);
      answerEl.lastElementChild.append(stats);
      const actions = document.createElement('div'); actions.className = 'message-actions';
      actions.innerHTML = `<button class="text-button copy-answer">Copy</button><button class="text-button save-answer">Save as note</button>`;
      answerEl.lastElementChild.append(actions);
    }
    // A return to this notebook replaces the detached live message with persisted history.
    if (state.current === notebookId) {
      try {
        const refreshEpoch = state.viewEpoch;
        const selection = state.selecting;
        if (selection?.id === notebookId && selection.viewEpoch === refreshEpoch) await selection.promise;
        if (state.current !== notebookId || state.viewEpoch !== refreshEpoch) return;
        const messages = await api(`/api/notebooks/${notebookId}/messages`);
        if (state.current === notebookId && state.viewEpoch === refreshEpoch) {
          state.messages = messages;
          if (!originalView()) renderMessages();
        }
      } catch (error) { toast(`Answer completed in ${title}; refresh to view history: ${error.message}`); }
    }
  } catch (error) {
    const failure = error.name === 'AbortError' ? 'Generation interrupted; refresh to check saved output.'
      : ['citation_validation', 'source_completeness'].includes(error.reason) ? error.message : `Generation failed: ${error.message}`;
    if (originalView()) {
      if (answer.trim()) rejectedDrafts.push(answer);
      const visibleDraft = rejectedDrafts.length > 1
        ? rejectedDrafts.map((draft, index) => `### Draft ${index + 1}\n\n${draft}`).join('\n\n')
        : rejectedDrafts[0] || 'No answer text was returned.';
      updateMessage(answerEl, visibleDraft, citations, false);
      const note = document.createElement('div'); note.className = 'rejected-answer-note';
      note.textContent = `${failure} ${rejectedDrafts.length ? 'The draft above was not saved.' : 'No answer was saved.'}`;
      answerEl.lastElementChild.append(note);
      const stats = document.createElement('div'); stats.className = 'answer-metrics';
      stats.textContent = responseMetrics(null, Date.now() - run.startedAt,
                                          run.firstTokenAt == null ? null : run.firstTokenAt - run.startedAt);
      answerEl.lastElementChild.append(stats);
    }
    if (error.name === 'AbortError') toast(`Generation interrupted in ${title}; refresh to view history.`);
    else toast(`${title}: ${error.message}`);
  } finally {
    if (state.run === run) { state.generating = false; state.run = null; renderGenerationStatus(); }
  }
}

function promptModal({title, nameLabel='Title', name='', textLabel=null, submit='Create'}) {
  const dialog = $('#promptDialog'); $('#promptTitle').textContent = title; $('#promptNameLabel').childNodes[0].nodeValue = nameLabel;
  $('#promptName').value = name; $('#promptTextLabel').classList.toggle('hidden', !textLabel); $('#promptTextLabel').childNodes[0].nodeValue = textLabel || 'Text'; $('#promptText').value = ''; $('#promptSubmit').textContent = submit;
  dialog.showModal(); setTimeout(() => $('#promptName').focus(), 30);
  return new Promise(resolve => {
    const finish = () => { dialog.removeEventListener('close', finish); resolve(dialog.returnValue === 'default' ? {name:$('#promptName').value.trim(), text:$('#promptText').value} : null); };
    dialog.addEventListener('close', finish);
  });
}

function confirmModal({title='Confirm action', message='Are you sure?', submit='Delete'}) {
  const dialog = $('#confirmDialog'); $('#confirmTitle').textContent = title; $('#confirmMessage').textContent = message; $('#confirmSubmit').textContent = submit;
  dialog.showModal();
  return new Promise(resolve => {
    const finish = () => { dialog.removeEventListener('close', finish); resolve(dialog.returnValue === 'confirm'); };
    dialog.addEventListener('close', finish);
  });
}

function notebookActionsModal(title) {
  const dialog = $('#notebookActionsDialog');
  $('#notebookActionsTitle').textContent = title;
  dialog.showModal();
  return new Promise(resolve => {
    const finish = () => { dialog.removeEventListener('close', finish); resolve(dialog.returnValue); };
    dialog.addEventListener('close', finish);
  });
}

async function createNotebook() {
  const result = await promptModal({title:'Create new notebook', nameLabel:'Notebook Title', submit:'Create'}); if (!result?.name) return;
  try { const item = await api('/api/notebooks', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:result.name})}); await loadNotebooks(item.id); toast('Notebook created.'); }
  catch (error) { toast(error.message); }
}

async function pasteSource() {
  const result = await promptModal({title:'Copied text', nameLabel:'Source Name', name:'Pasted text', textLabel:'Source Text', submit:'Add source'}); if (!result?.name || !result.text.trim()) return;
  try { await api(`/api/notebooks/${state.current}/sources/paste`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(result)}); state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling(); toast('Source added.'); }
  catch (error) { toast(error.message); }
}

async function uploadFiles(files) {
  for (const file of files) {
    const form = new FormData(); form.append('file', file);
    try {
      await api(`/api/notebooks/${state.current}/sources`, {method:'POST', body:form});
      toast(`Uploaded ${file.name}`);
    }
    catch (error) { toast(`${file.name}: ${error.message}`); }
  }
  state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling();
}

function renderSaved() {
  if (!state.detail) return;
  const items = [...state.detail.artifacts.map(x => ({...x, type:'artifact'})), ...state.detail.notes.map(x => ({...x, type:'note'}))]
    .sort((a,b) => b.updated_at.localeCompare(a.updated_at));

  $('#studioEmpty').classList.toggle('hidden', items.length > 0);
  $('#savedList').innerHTML = items.map(x => `
    <button class="saved-item" data-saved-type="${x.type}" data-saved-id="${x.id}">
      <b>${escapeHtml(x.title)}</b>
      <small>${escapeHtml(x.content.slice(0, 80))}${x.content.length > 80 ? '...' : ''}</small>
    </button>`).join('');
}

async function generateArtifact(kind, button) {
  if (state.generating) return toast('Finish current generation first.');
  if (!state.current) return;
  if (state.detail?.id !== state.current) return toast('Notebook is still loading.');
  const notebookId = state.current, title = state.detail.title;
  button.classList.add('loading'); state.generating = true;
  const controller = new AbortController();
  const run = {notebookId, title, kind:'studio', status:'Preparing source summaries, then generating Studio output…', controller};
  state.run = run; renderGenerationStatus();
  try {
    await streamRequest(`/api/notebooks/${notebookId}/artifacts`, {kind, source_ids:selectedSources()}, event => {
      if (event.type === 'status' && typeof event.message === 'string') { run.status = `${event.message} Generating source summaries, then final output…`; renderGenerationStatus(); }
      if (event.type === 'done' && event.metrics) {
        const {prompt_eval_count, eval_count} = event.metrics;
        if (event.num_ctx) state.numCtx = event.num_ctx;
        if (Number.isInteger(prompt_eval_count) && Number.isInteger(eval_count)) {
          updateContextMeter(prompt_eval_count, eval_count);
        }
      }
    }, controller.signal);
    toast(`Saved to Studio in ${title}.`);
    if (state.current === notebookId) {
      try {
        const viewEpoch = state.viewEpoch;
        const selection = state.selecting;
        if (selection?.id === notebookId && selection.viewEpoch === viewEpoch) await selection.promise;
        if (state.current !== notebookId || state.viewEpoch !== viewEpoch) return;
        const detail = await api(`/api/notebooks/${notebookId}`);
        if (state.current === notebookId && state.viewEpoch === viewEpoch) { state.detail = detail; renderSaved(); }
      } catch (error) { toast(`Refresh ${title} to view the saved artifact: ${error.message}`); }
    }
  } catch (error) { toast(error.name === 'AbortError' ? `Generation interrupted in ${title}; refresh to check saved output.` : `${title}: ${error.message}`); }
  finally { if (state.run === run) { state.generating = false; state.run = null; renderGenerationStatus(); } button.classList.remove('loading'); }
}

function openSaved(type, id) {
  const item = type === 'note' ? state.detail.notes.find(x => x.id === id) : state.detail.artifacts.find(x => x.id === id);
  if (!item) return; state.editing = {type, ...item}; $('#editorTitle').value = item.title; $('#editorContent').value = item.content; $('#deleteSaved').classList.remove('hidden'); $('#editorDialog').showModal();
}

async function saveCurrentEditor() {
  const body = {title:$('#editorTitle').value.trim(), content:$('#editorContent').value}; if (!body.title || !state.editing) return;
  const path = state.editing.type === 'note' ? `/api/notes/${state.editing.id}` : `/api/artifacts/${state.editing.id}`;
  await api(path, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  state.detail = await api(`/api/notebooks/${state.current}`); renderSaved(); toast('Saved.');
}

function resizeComposer() { const el=$('#question'); el.style.height='auto'; el.style.height=`${Math.min(el.scrollHeight,140)}px`; }

// Setup Drag & Drop File Upload over Left Sidebar
function setupDragAndDrop() {
  const dropZone = $('#dropZone');
  const panel = $('#sourcesPanel');
  if (!dropZone || !panel) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    panel.addEventListener(eventName, e => { e.preventDefault(); dropZone.classList.remove('hidden'); }, false);
  });
  ['dragleave', 'drop'].forEach(eventName => {
    panel.addEventListener(eventName, e => { e.preventDefault(); dropZone.classList.add('hidden'); }, false);
  });
  panel.addEventListener('drop', e => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0 && state.current) {
      uploadFiles([...dt.files]);
    }
  }, false);
}

document.addEventListener('click', async event => {
  const retryBtn = event.target.closest('[data-source-retry]');
  if (retryBtn) {
    event.stopPropagation();
    await api(`/api/sources/${retryBtn.dataset.sourceRetry}/retry`, {method:'POST'});
    state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling();
    toast('Source queued for processing.'); return;
  }
  const deleteBtn = event.target.closest('[data-source-delete]');
  if (deleteBtn) {
    event.stopPropagation();
    const id = deleteBtn.dataset.sourceDelete;
    const source = state.detail.sources.find(x => x.id === id);
    if (!source) return;
    const confirmDel = await confirmModal({
      title: 'Remove Source',
      message: `Delete "${source.name}" from this notebook?`,
      submit: 'Delete'
    });
    if (confirmDel) {
      await api(`/api/sources/${id}`, {method:'DELETE'});
      state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling();
      toast('Source deleted.');
    }
    return;
  }

  const sourceDetailsBtn = event.target.closest('[data-source-details]');
  if (sourceDetailsBtn) {
    const source = state.detail.sources.find(x => x.id === sourceDetailsBtn.dataset.sourceDetails);
    if (source) {
      $('#citationTitle').textContent = source.name;
      $('#citationLocation').textContent = `${source.kind.toUpperCase()}${source.page_count ? ` · ${source.page_count} pages` : ''}${source.char_count ? ` · ${Math.round(source.char_count/1000)}k chars` : ''} · ${source.status}`;
      $('#citationExcerpt').textContent = source.error || `Source file is processed and active. Included in Gemma local research grounding.`;
      $('#citationFile').href = `/api/files/${source.id}`;
      $('#citationDialog').showModal();
    }
    return;
  }

  const menuBtn = event.target.closest('[data-notebook-menu]');
  if (menuBtn) {
    event.stopPropagation();
    const id = menuBtn.dataset.notebookMenu;
    const item = state.notebooks.find(x => x.id === id);
    if (!item) return;
    const action = await notebookActionsModal(item.title);
    if (action === 'rename') {
      const result = await promptModal({title:'Rename notebook', nameLabel:'Notebook Title', name:item.title, submit:'Save'});
      if (result?.name) {
        await api(`/api/notebooks/${id}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:result.name})});
        await loadNotebooks(state.current === id ? id : undefined); toast('Notebook renamed.');
      }
    } else if (action === 'delete' && await confirmModal({
      title:'Delete Notebook', message:`Permanently delete "${item.title}" and all its sources?`, submit:'Delete'
    })) {
      await api(`/api/notebooks/${id}`, {method:'DELETE'});
      if (state.current === id) showLandingView();
      await loadNotebooks();
      toast('Notebook deleted.');
    }
    return;
  }

  const notebookCard = event.target.closest('[data-notebook]');
  if (notebookCard) return selectNotebook(notebookCard.dataset.notebook);

  const citation = event.target.closest('[data-citation]');
  if (citation) {
    const message = citation.closest('.message'), item = message._citations[Number(citation.dataset.citation)]; if (!item) return;
    $('#citationTitle').textContent = item.source_name;
    $('#citationLocation').textContent = item.page ? `Page ${item.page}` : 'Source excerpt';
    $('#citationExcerpt').textContent = item.excerpt;
    $('#citationFile').href = `/api/files/${item.source_id}${item.page ? `#page=${item.page}` : ''}`;
    $('#citationDialog').showModal(); return;
  }

  const suggestion = event.target.closest('.suggestion-chip'); if (suggestion) return sendQuestion(suggestion.textContent);

  const copy = event.target.closest('.copy-answer'); if (copy) {
    await navigator.clipboard.writeText(copy.closest('.message')._content); toast('Copied.'); return;
  }

  const saveAnswer = event.target.closest('.save-answer'); if (saveAnswer) {
    const message = saveAnswer.closest('.message');
    await api(`/api/notebooks/${state.current}/notes`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:'Saved Answer',content:message._content})});
    state.detail = await api(`/api/notebooks/${state.current}`); renderSaved(); toast('Saved as note.'); return;
  }

  const saved = event.target.closest('[data-saved-id]'); if (saved) return openSaved(saved.dataset.savedType, saved.dataset.savedId);
  const artifact = event.target.closest('[data-kind]'); if (artifact) return generateArtifact(artifact.dataset.kind, artifact);
});

document.addEventListener('change', async event => {
  if (event.target.matches('[data-toggle-source]')) {
    await api(`/api/sources/${event.target.dataset.toggleSource}/toggle`, {method:'PATCH'});
    state.detail = await api(`/api/notebooks/${state.current}`); renderSources();
  }
});

// Left Sidebar Search & Select All
$('#sourceSearchInput').oninput = event => {
  state.sourceQuery = event.target.value;
  renderSources();
};

$('#selectAllSourcesBtn').onclick = async () => {
  if (!state.detail?.sources.length) return;
  const readySources = state.detail.sources.filter(s => s.status === 'ready');
  const allEnabled = readySources.every(s => s.enabled);
  const targetValue = !allEnabled;

  for (const s of readySources) {
    if (s.enabled !== targetValue) {
      await api(`/api/sources/${s.id}/toggle`, {method:'PATCH'});
    }
  }
  state.detail = await api(`/api/notebooks/${state.current}`); renderSources();
  toast(targetValue ? 'All sources enabled.' : 'All sources disabled.');
};

// Header Events
$('#brandLogo').onclick = (e) => { e.preventDefault(); showLandingView(); };
$('#topCreateBtn').onclick = createNotebook;

// Add Source Modal Handlers
$('#addSourcesBtn').onclick = () => $('#addSourceModal').showModal();
$('#closeAddSourceModal').onclick = () => $('#addSourceModal').close();
$('#optionUploadFiles').onclick = () => { $('#addSourceModal').close(); $('#fileInput').click(); };
$('#optionPasteText').onclick = () => { $('#addSourceModal').close(); pasteSource(); };
$('#fileInput').onchange = event => { uploadFiles([...event.target.files]); event.target.value=''; };

// Chat Form
$('#chatForm').onsubmit = event => { event.preventDefault(); sendQuestion($('#question').value); };
$('#question').oninput = resizeComposer;
$('#question').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendQuestion(event.target.value); } };
$('#stopButton').onclick = () => { if (state.run?.kind === 'chat') state.run.controller.abort(); };
$('#studioStopButton').onclick = () => { if (state.run?.kind === 'studio') state.run.controller.abort(); };
$('#globalStopButton').onclick = () => state.run?.controller.abort();
$('#clearChat').onclick = async () => {
  const confirmClear = await confirmModal({title:'Clear Chat', message:'Clear all message history in this notebook?', submit:'Clear'});
  if (confirmClear) {
    await api(`/api/notebooks/${state.current}/messages`, {method:'DELETE'});
    state.messages=[]; renderMessages(); toast('Chat cleared.');
  }
};

// Studio Note
$('#newNote').onclick = async () => {
  const note = await api(`/api/notebooks/${state.current}/notes`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:'New Note',content:''})});
  state.detail.notes.unshift(note); renderSaved(); openSaved('note',note.id);
};

$('#editorForm').addEventListener('submit', async event => {
  event.preventDefault();
  try { await saveCurrentEditor(); $('#editorDialog').close('default'); } catch(error){toast(error.message);}
});

$('#deleteSaved').onclick = async () => {
  if (!state.editing) return;
  const confirmDel = await confirmModal({title:'Delete Item', message:`Delete "${state.editing.title}"?`, submit:'Delete'});
  if (!confirmDel) return;
  const path=state.editing.type==='note'?`/api/notes/${state.editing.id}`:`/api/artifacts/${state.editing.id}`;
  await api(path,{method:'DELETE'}); $('#editorDialog').close('cancel');
  state.detail=await api(`/api/notebooks/${state.current}`); renderSaved(); toast('Deleted.');
};

// Notebook Title Rename
$('#notebookTitleBtn').onclick = async () => {
  if (!state.current) return;
  const currentItem = state.notebooks.find(x => x.id === state.current);
  const renameResult = await promptModal({
    title: 'Rename Notebook',
    nameLabel: 'Notebook Title',
    name: currentItem ? currentItem.title : '',
    submit: 'Save'
  });
  if (renameResult?.name) {
    await api(`/api/notebooks/${state.current}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:renameResult.name})});
    await loadNotebooks(state.current);
    toast('Renamed.');
  }
};

function setTheme(dark) {
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  $('.theme-icon-dark').classList.toggle('hidden', dark);
  $('.theme-icon-light').classList.toggle('hidden', !dark);
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}

$('#themeButton').onclick = () => {
  const isDark = document.documentElement.dataset.theme === 'dark';
  setTheme(!isDark);
};

$('#sourcesClose').onclick = () => $('#sourcesPanel').classList.remove('open');
$('#studioToggle').onclick = () => $('#studioPanel').classList.add('open');
$('#studioClose').onclick = () => $('#studioPanel').classList.remove('open');

$('#modelSelect').addEventListener('change', event => changeSettings({generation_model: event.target.value}));
$('#ctxSelect').addEventListener('change', event => changeSettings({num_ctx: Number(event.target.value)}));

// Initialize
const initialDark = localStorage.getItem('theme') ? localStorage.getItem('theme') === 'dark' : true;
setTheme(initialDark);
setupDragAndDrop();
loadHealth();
loadSettings();
loadNotebooks().catch(error => toast(error.message));
setInterval(loadHealth, 30000);
