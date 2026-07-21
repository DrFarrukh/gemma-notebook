const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { notebooks: [], current: null, detail: null, messages: [], generating: false, controller: null, editing: null, poller: null };

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
  let safe = escapeHtml(value);
  safe = safe.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>');
  safe = safe.replace(/^### (.+)$/gm, '<h3>$1</h3>').replace(/^## (.+)$/gm, '<h2>$1</h2>').replace(/^# (.+)$/gm, '<h1>$1</h1>');
  safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/\[(\d+)\]/g, (_, n) => citations[Number(n)-1] ? `<button class="citation" data-citation="${Number(n)-1}">${n}</button>` : `[${n}]`);
  const blocks = safe.split(/\n{2,}/).map(block => {
    if (/^<(h\d|pre)/.test(block)) return block;
    const lines = block.split('\n');
    if (lines.every(line => /^[-*] /.test(line))) return `<ul>${lines.map(line => `<li>${line.slice(2)}</li>`).join('')}</ul>`;
    return `<p>${block.replace(/\n/g, '<br>')}</p>`;
  });
  return blocks.join('');
}

function selectedSources() {
  return (state.detail?.sources || []).filter(s => s.enabled && s.status === 'ready').map(s => s.id);
}

async function loadHealth() {
  try {
    const health = await api('/api/health');
    const el = $('#modelStatus');
    if (health.ollama && health.generation_ready && health.embedding_ready) {
      el.className = 'status good'; el.querySelector('span').textContent = 'Local models ready';
    } else {
      el.className = 'status bad'; el.querySelector('span').textContent = health.ollama ? 'Model missing' : 'Ollama offline';
    }
    el.title = `${health.generation_model} · ${health.embedding_model}${health.error ? ` · ${health.error}` : ''}`;
  } catch (error) { $('#modelStatus').className = 'status bad'; }
}

async function loadNotebooks(selectId) {
  state.notebooks = await api('/api/notebooks');
  renderNotebooks();
  const target = selectId || state.current || state.notebooks[0]?.id;
  if (target && state.notebooks.some(n => n.id === target)) await selectNotebook(target);
  else showWelcome();
}

function renderNotebooks() {
  $('#notebookList').innerHTML = state.notebooks.map(n => `
    <div class="notebook-row ${n.id === state.current ? 'active' : ''}">
      <button class="notebook-item" data-notebook="${n.id}"><span>◫</span><div><b>${escapeHtml(n.title)}</b><small>${n.source_count || 0} sources</small></div></button>
      <button class="notebook-more" data-notebook-menu="${n.id}" title="Rename or delete">•••</button>
    </div>`).join('') || '<div class="studio-empty">No notebooks yet.</div>';
}

function showWelcome() {
  state.current = null; state.detail = null;
  $('#workspace').classList.add('empty'); $('#welcome').classList.remove('hidden'); $('#chatContent').classList.add('hidden');
  $('#sourcesSection').classList.add('hidden'); $('#studioContent').classList.add('hidden'); $('#studioEmpty').classList.remove('hidden');
  $('#currentTitle').textContent = 'Choose a notebook';
}

async function selectNotebook(id) {
  state.current = id;
  [state.detail, state.messages] = await Promise.all([api(`/api/notebooks/${id}`), api(`/api/notebooks/${id}/messages`)]);
  $('#workspace').classList.remove('empty'); $('#welcome').classList.add('hidden'); $('#chatContent').classList.remove('hidden');
  $('#sourcesSection').classList.remove('hidden'); $('#studioContent').classList.remove('hidden'); $('#studioEmpty').classList.add('hidden');
  $('#currentTitle').textContent = state.detail.title;
  renderNotebooks(); renderSources(); renderMessages(); renderSaved(); updateSuggestions();
  $('#sourcesPanel').classList.remove('open');
  schedulePolling();
}

function renderSources() {
  const sources = state.detail?.sources || [];
  $('#sourceCount').textContent = sources.length;
  const ready = sources.filter(s => s.status === 'ready' && s.enabled).length;
  $('#readySourceCount').textContent = ready;
  $('#selectedSourceLabel').textContent = `${ready} enabled source${ready === 1 ? '' : 's'}`;
  $('#sourceList').innerHTML = sources.map(s => `
    <div class="source-item">
      <input type="checkbox" data-toggle-source="${s.id}" ${s.enabled ? 'checked' : ''} ${s.status !== 'ready' ? 'disabled' : ''} title="Include in research">
      <div class="source-info"><b title="${escapeHtml(s.name)}">${escapeHtml(s.name)}</b>
        <small>${s.kind.toUpperCase()}${s.page_count ? ` · ${s.page_count} pages` : ''}${s.char_count ? ` · ${Math.round(s.char_count/1000)}k chars` : ''}</small>
        <span class="status-pill ${s.status}">${s.status}</span>
        ${s.error ? `<small class="error">${escapeHtml(s.error)}</small>` : ''}
      </div>
      <button class="source-menu" data-source-menu="${s.id}" title="${s.status === 'error' ? 'Retry or delete' : 'Delete source'}">•••</button>
    </div>`).join('') || '<div class="studio-empty">Upload files or paste text to begin.</div>';
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
  $('#chatIntro').classList.toggle('hidden', state.messages.length > 0);
  area.querySelectorAll('.message').forEach(el => el.remove());
  state.messages.forEach(message => appendMessage(message.role, message.content, message.citations || [], false));
  area.scrollTop = area.scrollHeight;
}

function appendMessage(role, content, citations = [], streaming = false) {
  const el = document.createElement('article'); el.className = `message ${role}`;
  el.innerHTML = `<div class="avatar">${role === 'assistant' ? 'G' : 'You'}</div><div><div class="message-body ${streaming ? 'typing' : ''}">${markdown(content, citations)}</div>${role === 'assistant' && !streaming ? `<div class="message-actions"><button class="text-button copy-answer">Copy</button><button class="text-button save-answer">Save as note</button></div>` : ''}</div>`;
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
    ? ['What are the main themes across these sources?', 'Where do the sources agree or disagree?', 'What important questions remain unanswered?']
    : ['Summarize the key ideas in this source.', 'What evidence supports the main argument?', 'Create five questions to test my understanding.'];
  $('#suggestions').innerHTML = options.map(x => `<button class="suggestion">${escapeHtml(x)}</button>`).join('');
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
  state.generating = true; $('#chatIntro').classList.add('hidden'); $('#question').value = ''; resizeComposer();
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
    const actions = document.createElement('div'); actions.className = 'message-actions'; actions.innerHTML = '<button class="text-button copy-answer">Copy</button><button class="text-button save-answer">Save as note</button>';
    answerEl.lastElementChild.append(actions);
    state.messages = await api(`/api/notebooks/${state.current}/messages`);
  } catch (error) {
    if (error.name === 'AbortError') updateMessage(answerEl, answer || '_Generation stopped._', citations, false);
    else { updateMessage(answerEl, answer || `Error: ${error.message}`, citations, false); toast(error.message); }
  } finally {
    state.generating = false; state.controller = null; $('#sendButton').classList.remove('hidden'); $('#stopButton').classList.add('hidden');
  }
}

function promptModal({title, nameLabel='Name', name='', textLabel=null, submit='Create'}) {
  const dialog = $('#promptDialog'); $('#promptTitle').textContent = title; $('#promptNameLabel').childNodes[0].nodeValue = nameLabel;
  $('#promptName').value = name; $('#promptTextLabel').classList.toggle('hidden', !textLabel); $('#promptTextLabel').childNodes[0].nodeValue = textLabel || 'Text'; $('#promptText').value = ''; $('#promptSubmit').textContent = submit;
  dialog.showModal(); setTimeout(() => $('#promptName').focus(), 30);
  return new Promise(resolve => {
    const finish = () => { dialog.removeEventListener('close', finish); resolve(dialog.returnValue === 'default' ? {name:$('#promptName').value.trim(), text:$('#promptText').value} : null); };
    dialog.addEventListener('close', finish);
  });
}

async function createNotebook() {
  const result = await promptModal({title:'New notebook', nameLabel:'Notebook name', submit:'Create notebook'}); if (!result?.name) return;
  try { const item = await api('/api/notebooks', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:result.name})}); await loadNotebooks(item.id); }
  catch (error) { toast(error.message); }
}

async function pasteSource() {
  const result = await promptModal({title:'Paste a source', nameLabel:'Source name', name:'Pasted text', textLabel:'Source text', submit:'Add source'}); if (!result?.name || !result.text.trim()) return;
  try { await api(`/api/notebooks/${state.current}/sources/paste`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(result)}); state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling(); }
  catch (error) { toast(error.message); }
}

async function uploadFiles(files) {
  for (const file of files) {
    const form = new FormData(); form.append('file', file);
    try { await api(`/api/notebooks/${state.current}/sources`, {method:'POST', body:form}); }
    catch (error) { toast(`${file.name}: ${error.message}`); }
  }
  state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling();
}

function renderSaved() {
  if (!state.detail) return;
  const items = [...state.detail.artifacts.map(x => ({...x, type:'artifact'})), ...state.detail.notes.map(x => ({...x, type:'note'}))]
    .sort((a,b) => b.updated_at.localeCompare(a.updated_at));
  $('#savedList').innerHTML = items.map(x => `<button class="saved-item" data-saved-type="${x.type}" data-saved-id="${x.id}"><span class="kind">${x.type === 'note' ? 'Note' : escapeHtml(x.kind)}</span><b>${escapeHtml(x.title)}</b><small>${new Date(x.updated_at).toLocaleString()}</small></button>`).join('') || '<div class="studio-empty">Nothing saved yet.</div>';
}

async function generateArtifact(kind, button) {
  if (state.generating) return toast('Finish the current generation first.');
  button.classList.add('loading'); state.generating = true; let content = '', citations = [];
  const title = {summary:'Generating summary…',faq:'Generating FAQ…',guide:'Generating study guide…'}[kind];
  state.editing = {type:'artifact-live', id:null, title, content:''}; $('#editorTitle').value = title; $('#editorContent').value = ''; $('#deleteSaved').classList.add('hidden'); $('#editorDialog').showModal();
  try {
    await streamRequest(`/api/notebooks/${state.current}/artifacts`, {kind, source_ids:selectedSources()}, event => {
      if (event.type === 'citations') citations = event.citations;
      if (event.type === 'delta') { content += event.text; $('#editorContent').value = content; }
      if (event.type === 'error') throw new Error(event.message);
    });
    $('#editorDialog').close('cancel'); state.detail = await api(`/api/notebooks/${state.current}`); renderSaved(); toast('Studio artifact saved.');
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
  state.detail = await api(`/api/notebooks/${state.current}`); renderSaved();
}

function resizeComposer() { const el=$('#question'); el.style.height='auto'; el.style.height=`${Math.min(el.scrollHeight,160)}px`; }

document.addEventListener('click', async event => {
  const notebookMenu = event.target.closest('[data-notebook-menu]'); if (notebookMenu) {
    const item = state.notebooks.find(x => x.id === notebookMenu.dataset.notebookMenu); if (!item) return;
    const action = prompt(`Notebook “${item.title}”\nType R to rename or D to delete.`)?.trim().toLowerCase();
    if (action === 'r') {
      const result = await promptModal({title:'Rename notebook',nameLabel:'Notebook name',name:item.title,submit:'Rename'});
      if (result?.name) { await api(`/api/notebooks/${item.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:result.name})}); await loadNotebooks(item.id); }
    } else if (action === 'd' && confirm(`Permanently delete “${item.title}”, its sources, chat, and notes?`)) {
      await api(`/api/notebooks/${item.id}`,{method:'DELETE'}); if (state.current === item.id) state.current = null; await loadNotebooks();
    }
    return;
  }
  const notebook = event.target.closest('[data-notebook]'); if (notebook) return selectNotebook(notebook.dataset.notebook);
  const citation = event.target.closest('[data-citation]'); if (citation) {
    const message = citation.closest('.message'), item = message._citations[Number(citation.dataset.citation)]; if (!item) return;
    $('#citationTitle').textContent = item.source_name; $('#citationLocation').textContent = item.page ? `Page ${item.page}` : 'Source excerpt'; $('#citationExcerpt').textContent = item.excerpt; $('#citationFile').href = `/api/files/${item.source_id}`; $('#citationDialog').showModal(); return;
  }
  const suggestion = event.target.closest('.suggestion'); if (suggestion) return sendQuestion(suggestion.textContent);
  const sourceMenu = event.target.closest('[data-source-menu]'); if (sourceMenu) {
    const source = state.detail.sources.find(x => x.id === sourceMenu.dataset.sourceMenu); if (!source) return;
    if (source.status === 'error') {
      const action = prompt(`Source “${source.name}” failed.\nType R to retry or D to delete.`)?.trim().toLowerCase();
      if (action === 'r') await api(`/api/sources/${source.id}/retry`, {method:'POST'});
      if (action === 'd' && confirm(`Delete “${source.name}”?`)) await api(`/api/sources/${source.id}`, {method:'DELETE'});
    } else if (confirm(`Delete “${source.name}” and its index?`)) await api(`/api/sources/${source.id}`, {method:'DELETE'});
    state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); schedulePolling(); return;
  }
  const copy = event.target.closest('.copy-answer'); if (copy) { await navigator.clipboard.writeText(copy.closest('.message')._content); toast('Copied.'); return; }
  const saveAnswer = event.target.closest('.save-answer'); if (saveAnswer) {
    const message = saveAnswer.closest('.message'); const note = await api(`/api/notebooks/${state.current}/notes`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:'Saved answer',content:message._content})}); state.detail = await api(`/api/notebooks/${state.current}`); renderSaved(); toast('Saved as a note.'); return;
  }
  const saved = event.target.closest('[data-saved-id]'); if (saved) return openSaved(saved.dataset.savedType, saved.dataset.savedId);
  const artifact = event.target.closest('[data-kind]'); if (artifact) return generateArtifact(artifact.dataset.kind, artifact);
});

document.addEventListener('change', async event => {
  if (event.target.matches('[data-toggle-source]')) { await api(`/api/sources/${event.target.dataset.toggleSource}/toggle`, {method:'PATCH'}); state.detail = await api(`/api/notebooks/${state.current}`); renderSources(); }
});

$('#newNotebook').onclick = $('#welcomeCreate').onclick = createNotebook;
$('#pasteButton').onclick = pasteSource; $('#uploadButton').onclick = () => $('#fileInput').click();
$('#fileInput').onchange = event => { uploadFiles([...event.target.files]); event.target.value=''; };
$('#chatForm').onsubmit = event => { event.preventDefault(); sendQuestion($('#question').value); };
$('#question').oninput = resizeComposer; $('#question').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendQuestion(event.target.value); } };
$('#stopButton').onclick = () => state.controller?.abort();
$('#clearChat').onclick = async () => { if (confirm('Clear this notebook’s chat history?')) { await api(`/api/notebooks/${state.current}/messages`, {method:'DELETE'}); state.messages=[]; renderMessages(); } };
$('#newNote').onclick = async () => { const note = await api(`/api/notebooks/${state.current}/notes`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:'New note',content:''})}); state.detail.notes.unshift(note); renderSaved(); openSaved('note',note.id); };
$('#editorForm').addEventListener('submit', async event => { event.preventDefault(); if (state.editing?.type === 'artifact-live') return; try { await saveCurrentEditor(); $('#editorDialog').close('default'); } catch(error){toast(error.message);} });
$('#deleteSaved').onclick = async () => { if (!state.editing || !confirm(`Delete “${state.editing.title}”?`)) return; const path=state.editing.type==='note'?`/api/notes/${state.editing.id}`:`/api/artifacts/${state.editing.id}`; await api(path,{method:'DELETE'}); $('#editorDialog').close('cancel'); state.detail=await api(`/api/notebooks/${state.current}`); renderSaved(); };
$('#themeButton').onclick = () => { const dark=document.documentElement.dataset.theme==='dark'; document.documentElement.dataset.theme=dark?'light':'dark'; localStorage.setItem('theme',dark?'light':'dark'); };
$('#sourcesToggle').onclick = () => $('#sourcesPanel').classList.toggle('open'); $('#studioToggle').onclick = () => $('#studioPanel').classList.add('open'); $('#studioClose').onclick = () => $('#studioPanel').classList.remove('open');

document.documentElement.dataset.theme = localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
loadHealth(); loadNotebooks().catch(error => toast(error.message)); setInterval(loadHealth, 30000);
