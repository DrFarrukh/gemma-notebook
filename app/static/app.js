const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  notebooks: [],
  current: null,
  detail: null,
  messages: [],
  generating: false,
  controller: null,
  editing: null,
  poller: null,
  sourceQuery: ''
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

function escapeHtml(value = '') {
  return value.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function markdown(value, citations = []) {
  const inline = text => escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(\d+)\]/g, (_, n) => citations[Number(n)-1]
      ? `<button class="citation" data-citation="${Number(n)-1}" title="Open source ${n}">${n}</button>`
      : `[${n}]`);
  const output = [];
  let list = null, paragraph = [], code = [], inCode = false;
  const closeParagraph = () => { if (paragraph.length) output.push(`<p>${paragraph.join('<br>')}</p>`); paragraph = []; };
  const closeList = () => { if (list) output.push(`</${list}>`); list = null; };
  const lines = String(value || '').split('\n');
  const tableCells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
  const isTableDivider = line => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const rawLine = lines[lineIndex];
    if (/^\s*```/.test(rawLine)) {
      closeParagraph(); closeList();
      if (inCode) { output.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`); code = []; }
      inCode = !inCode; continue;
    }
    if (inCode) { code.push(rawLine); continue; }
    if (rawLine.includes('|') && lineIndex + 1 < lines.length && isTableDivider(lines[lineIndex + 1])) {
      closeParagraph(); closeList();
      const headings = tableCells(rawLine);
      const rows = [];
      lineIndex += 2;
      while (lineIndex < lines.length && lines[lineIndex].trim() && lines[lineIndex].includes('|')) {
        rows.push(tableCells(lines[lineIndex]));
        lineIndex++;
      }
      lineIndex--;
      const width = headings.length;
      const head = `<thead><tr>${headings.map(cell => `<th>${inline(cell)}</th>`).join('')}</tr></thead>`;
      const body = rows.length ? `<tbody>${rows.map(row => `<tr>${Array.from({length: width}, (_, index) => `<td>${inline(row[index] || '')}</td>`).join('')}</tr>`).join('')}</tbody>` : '';
      output.push(`<div class="table-wrap"><table>${head}${body}</table></div>`);
      continue;
    }
    const heading = rawLine.match(/^\s*(#{1,3})\s+(.+)$/);
    const unordered = rawLine.match(/^\s*[-*]\s+(.+)$/);
    const ordered = rawLine.match(/^\s*\d+[.)]\s+(.+)$/);
    if (/^\s*(\*{3,}|-{3,}|_{3,})\s*$/.test(rawLine)) { closeParagraph(); closeList(); output.push('<hr>'); }
    else if (heading) { closeParagraph(); closeList(); const level = heading[1].length; output.push(`<h${level}>${inline(heading[2])}</h${level}>`); }
    else if (unordered || ordered) {
      closeParagraph(); const wanted = unordered ? 'ul' : 'ol';
      if (list !== wanted) { closeList(); output.push(`<${wanted}>`); list = wanted; }
      output.push(`<li>${inline((unordered || ordered)[1])}</li>`);
    } else if (!rawLine.trim()) { closeParagraph(); closeList(); }
    else { closeList(); paragraph.push(inline(rawLine)); }
  }
  if (inCode) output.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
  closeParagraph(); closeList();
  return output.join('');
}

function selectedSources() {
  return (state.detail?.sources || []).filter(s => s.enabled && s.status === 'ready').map(s => s.id);
}

async function loadHealth() {
  try {
    const health = await api('/api/health');
    const el = $('#modelStatus');
    const label = el.querySelector('.status-label');
    if (health.ollama && health.generation_ready && health.embedding_ready) {
      el.className = 'status good';
      if (label) label.textContent = 'Gemma 4 Ready';
    } else {
      el.className = 'status bad';
      if (label) label.textContent = health.ollama ? 'Model missing' : 'Ollama offline';
    }
  } catch (error) { $('#modelStatus').className = 'status bad'; }
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
  state.current = null;
  state.detail = null;
  $('#landingView').classList.remove('hidden');
  $('#workspace').classList.add('hidden');
  $('#notebookNavTitle').classList.add('hidden');
  $('#landingNavTabs').classList.remove('hidden');
}

async function selectNotebook(id) {
  state.current = id;
  [state.detail, state.messages] = await Promise.all([api(`/api/notebooks/${id}`), api(`/api/notebooks/${id}/messages`)]);

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
  state.poller = setTimeout(async () => {
    if (!state.current) return;
    try { state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); renderSaved(); schedulePolling(); }
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

function updateMessage(el, content, citations, streaming) {
  el._content = content; el._citations = citations;
  const body = el.querySelector('.message-body'); body.innerHTML = markdown(content, citations); body.classList.toggle('typing', streaming);
  $('#messages').scrollTop = $('#messages').scrollHeight;
}

function updateSuggestions() {
  const sources = state.detail?.sources.filter(s => s.status === 'ready') || [];
  const options = sources.length > 1
    ? ['What are the main themes across these sources?', 'Where do the sources agree or disagree?', 'What key evidence is provided?']
    : ['Summarize the core ideas in this source.', 'What key evidence supports the main argument?', 'Extract key definitions and takeaways.'];
  $('#suggestions').innerHTML = options.map(x => `<button class="suggestion-chip">${escapeHtml(x)}</button>`).join('');
}

async function streamRequest(path, body, onEvent) {
  const controller = new AbortController(); state.controller = controller;
  const response = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body), signal:controller.signal });
  if (!response.ok) { let data; try { data = await response.json(); } catch (_) {} throw new Error(data?.detail || response.statusText); }
  const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
  while (true) {
    const {value, done} = await reader.read(); if (done) break;
    buffer += decoder.decode(value, {stream:true}); const lines = buffer.split('\n'); buffer = lines.pop();
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
  }
}

async function sendQuestion(question) {
  if (!state.current || state.generating || !question.trim()) return;
  state.generating = true; $('#notebookBanner').classList.add('hidden'); $('#question').value = ''; resizeComposer();
  appendMessage('user', question.trim());
  const answerEl = appendMessage('assistant', '', [], true); let answer = '', citations = [];
  $('#sendButton').classList.add('hidden'); $('#stopButton').classList.remove('hidden');
  try {
    await streamRequest(`/api/notebooks/${state.current}/chat`, {question:question.trim(), source_ids:selectedSources()}, event => {
      if (event.type === 'citations') citations = event.citations;
      if (event.type === 'delta') { answer += event.text; updateMessage(answerEl, answer, citations, true); }
      if (event.type === 'error') throw new Error(event.message);
    });
    updateMessage(answerEl, answer, citations, false);
    const actions = document.createElement('div'); actions.className = 'message-actions';
    actions.innerHTML = `<button class="text-button copy-answer">Copy</button><button class="text-button save-answer">Save as note</button>`;
    answerEl.lastElementChild.append(actions);
    state.messages = await api(`/api/notebooks/${state.current}/messages`);
  } catch (error) {
    if (error.name === 'AbortError') updateMessage(answerEl, answer || '_Generation stopped._', citations, false);
    else { updateMessage(answerEl, answer || `Error: ${error.message}`, citations, false); toast(error.message); }
  } finally {
    state.generating = false; state.controller = null; $('#sendButton').classList.remove('hidden'); $('#stopButton').classList.add('hidden');
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
  button.classList.add('loading'); state.generating = true; let content = '', citations = [];
  const titleMap = {summary:'Summary', faq:'FAQ', guide:'Study Guide'};
  const title = titleMap[kind] || 'Generated Artifact';
  state.editing = {type:'artifact-live', id:null, title, content:''};
  $('#editorTitle').value = title; $('#editorContent').value = ''; $('#deleteSaved').classList.add('hidden'); $('#editorDialog').showModal();
  try {
    await streamRequest(`/api/notebooks/${state.current}/artifacts`, {kind, source_ids:selectedSources()}, event => {
      if (event.type === 'citations') citations = event.citations;
      if (event.type === 'delta') { content += event.text; $('#editorContent').value = content; }
      if (event.type === 'error') throw new Error(event.message);
    });
    $('#editorDialog').close('cancel'); state.detail = await api(`/api/notebooks/${state.current}`); renderSaved(); toast('Saved to Studio.');
  } catch (error) { if (error.name !== 'AbortError') toast(error.message); }
  finally { state.generating = false; state.controller = null; button.classList.remove('loading'); }
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
$('#stopButton').onclick = () => state.controller?.abort();
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
  event.preventDefault(); if (state.editing?.type === 'artifact-live') return;
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

// Initialize
const initialDark = localStorage.getItem('theme') ? localStorage.getItem('theme') === 'dark' : true;
setTheme(initialDark);
setupDragAndDrop();
loadHealth();
loadNotebooks().catch(error => toast(error.message));
setInterval(loadHealth, 30000);
