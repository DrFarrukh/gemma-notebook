const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const {markdown, escapeHtml} = require('../app/static/markdown.js');

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function uiHarness() {
  const nodes = new Map(), requests = [], streams = [], toasts = [];
  function element() {
    const classes = new Set(['hidden']);
    return {
      classList: {
        add: key => classes.add(key), remove: key => classes.delete(key),
        toggle(key, on) { if (on === undefined) on = !classes.has(key); if (on) classes.add(key); else classes.delete(key); },
        contains: key => classes.has(key)
      },
      style: {}, scrollHeight: 0, scrollTop: 0, textContent: '', value: '', innerHTML: '',
      lastElementChild: {append() {}},
      append(child) { if (child.className === 'toast') toasts.push(child.textContent); },
      remove() {}, setAttribute(key, value) { this[key] = value; }, addEventListener() {}, querySelector() { return element(); }, querySelectorAll() { return []; }
    };
  }
  const node = selector => { if (!nodes.has(selector)) nodes.set(selector, element()); return nodes.get(selector); };
  const notebooks = {
    A: {id: 'A', title: 'Notebook A', updated_at: '2026-01-01', sources: [], notes: [], artifacts: []},
    B: {id: 'B', title: 'Notebook B', updated_at: '2026-01-01', sources: [], notes: [], artifacts: []}
  };
  const histories = {A: [], B: []};
  const pending = new Map();
  const context = vm.createContext({
    document: {documentElement: {dataset: {}}, querySelector: node, querySelectorAll: () => [], createElement: element, addEventListener() {}},
    localStorage: {getItem: () => null, setItem() {}}, setTimeout: () => 0, clearTimeout() {}, setInterval() {},
    AbortController, DOMException, markdown, escapeHtml,
    fetch: async path => {
      requests.push(path);
      if (pending.has(path)) {
        const response = pending.get(path);
        return {ok: true, json: async () => response.promise};
      }
      const body = path === '/api/health' ? {ollama: true, generation_ready: true, embedding_ready: true}
        : path === '/api/notebooks' ? Object.values(notebooks)
        : path.endsWith('/messages') ? histories[path.split('/')[3]] : notebooks[path.split('/').at(-1)];
      return {ok: true, json: async () => body};
    },
    streamRequest: (path, body, onEvent, signal) => {
      const gate = deferred();
      const stream = {path, body, onEvent, signal, resolve() { onEvent({type: 'done'}); gate.resolve(); }};
      streams.push(stream);
      signal.addEventListener('abort', () => gate.reject(new DOMException('Stopped', 'AbortError')), {once: true});
      return gate.promise;
    }
  });
  vm.runInContext(readFileSync(require.resolve('../app/static/app.js'), 'utf8'), context);
  const state = vm.runInContext('state', context);
  return {context, state, node, notebooks, histories, requests, streams, toasts, pending, ready: new Promise(setImmediate)};
}

test('Studio Stop is visible and works during both source-summary and final stages', async () => {
  for (const stage of ['map', 'final']) {
    const ui = uiHarness();
    await ui.ready;
    await ui.context.selectNotebook('A');
    const generation = ui.context.generateArtifact('summary', ui.node('#studioTool'));
    const stream = ui.streams[0];
    assert.equal(stream.path, '/api/notebooks/A/artifacts');
    assert.equal(ui.node('#studioStopButton').classList.contains('hidden'), false);
    if (stage === 'final') {
      stream.onEvent({type: 'status', message: 'Sampled evidence.'});
      stream.onEvent({type: 'delta', text: 'Synthesis started'});
      assert.match(ui.node('#studioStatus').textContent, /Generating source summaries, then final output/);
    }
    ui.node('#studioStopButton').onclick();
    await generation;
    assert.equal(stream.signal.aborted, true);
    assert.equal(ui.node('#studioStopButton').classList.contains('hidden'), true);
    assert.match(ui.toasts.at(-1), /refresh to check saved output/);
  }
});

test('Studio completes for its origin when switched to another notebook; stale refresh cannot overwrite new view', async () => {
  const ui = uiHarness();
  await ui.ready;
  await ui.context.selectNotebook('A');
  const generation = ui.context.generateArtifact('summary', ui.node('#studioTool'));
  const stream = ui.streams[0];
  await ui.context.selectNotebook('B');
  stream.onEvent({type: 'status', message: 'Sampled.'});
  assert.equal(ui.node('#studioStatus').textContent, 'Studio running in Notebook A…');
  stream.resolve();
  await generation;
  assert.equal(ui.state.detail.id, 'B');
  assert.equal(ui.toasts.at(-1), 'Saved to Studio in Notebook A.');
  assert.equal(ui.requests.filter(path => path === '/api/notebooks/B').length, 1);

  await ui.context.selectNotebook('A');
  const refresh = deferred(); ui.pending.set('/api/notebooks/A', refresh);
  const another = ui.context.generateArtifact('faq', ui.node('#studioTool'));
  ui.streams[1].resolve();
  await Promise.resolve(); await Promise.resolve();
  await ui.context.selectNotebook('B');
  refresh.resolve(ui.notebooks.A);
  await another;
  assert.equal(ui.state.detail.id, 'B');
});

test('chat stream remains scoped after navigation; a stale history refresh cannot replace the new notebook', async () => {
  const ui = uiHarness();
  await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('What happened?');
  const stream = ui.streams[0];
  assert.equal(stream.path, '/api/notebooks/A/chat');
  await ui.context.selectNotebook('B');
  stream.onEvent({type: 'delta', text: 'A answer'});
  assert.equal(ui.node('#chatStatus').textContent, 'Chat running in Notebook A…');
  stream.resolve(); await chat;
  assert.equal(ui.state.detail.id, 'B');
  assert.equal(ui.requests.filter(path => path === '/api/notebooks/B/messages').length, 1);

  await ui.context.selectNotebook('A');
  const refresh = deferred(); ui.pending.set('/api/notebooks/A/messages', refresh);
  const other = ui.context.sendQuestion('Second question');
  ui.streams[1].resolve();
  await Promise.resolve(); await Promise.resolve();
  await ui.context.selectNotebook('B');
  refresh.resolve([{role: 'assistant', content: 'A only'}]);
  await other;
  assert.equal(ui.state.detail.id, 'B');
  assert.equal(ui.state.messages.length, 0);
});

test('landing view retains a global Stop control for a running Studio job', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const generation = ui.context.generateArtifact('summary', ui.node('#studioTool'));
  ui.context.showLandingView();
  assert.equal(ui.node('#globalStopButton').classList.contains('hidden'), false);
  assert.equal(ui.node('#globalStopButton')['aria-label'], 'Stop generation in Notebook A');
  ui.node('#globalStopButton').onclick();
  await generation;
  assert.equal(ui.streams[0].signal.aborted, true);
  assert.equal(ui.node('#globalStopButton').classList.contains('hidden'), true);
});

test('chat Stop after changing notebook does not claim a server-side rollback', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('What happened?');
  await ui.context.selectNotebook('B');
  ui.node('#stopButton').onclick();
  await chat;
  assert.equal(ui.streams[0].signal.aborted, true);
  assert.equal(ui.state.detail.id, 'B');
  assert.match(ui.toasts.at(-1), /Notebook A; refresh to check saved output/);
});

test('late notebook selection responses cannot replace a newer view or start with old sources', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const response = deferred(); ui.pending.set('/api/notebooks/B', response);
  const loading = ui.context.selectNotebook('B');
  await ui.context.generateArtifact('faq', ui.node('#studioTool'));
  assert.equal(ui.streams.length, 0);
  assert.match(ui.toasts.at(-1), /still loading/);
  await ui.context.selectNotebook('A');
  response.resolve(ui.notebooks.B);
  await loading;
  assert.equal(ui.state.detail.id, 'A');
});

test('chat completion waits for the returning notebook selection before refreshing persisted history', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Question');
  await ui.context.selectNotebook('B');
  const oldHistory = deferred(); ui.pending.set('/api/notebooks/A/messages', oldHistory);
  const selecting = ui.context.selectNotebook('A');
  ui.streams[0].onEvent({type: 'delta', text: 'Saved answer'});
  ui.streams[0].resolve();
  await Promise.resolve(); await Promise.resolve();
  assert.equal(ui.requests.filter(path => path === '/api/notebooks/A/messages').length, 2); // initial A + pending selection
  ui.pending.delete('/api/notebooks/A/messages');
  ui.histories.A = [{role: 'assistant', content: 'Saved answer'}];
  oldHistory.resolve([]); // pre-commit snapshot finishes after done
  await selecting;
  await chat;
  assert.equal(ui.requests.filter(path => path === '/api/notebooks/A/messages').length, 3);
  assert.equal(ui.state.messages[0].content, 'Saved answer');
});

test('Studio completion waits for the returning notebook selection before refreshing saved artifacts', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const generation = ui.context.generateArtifact('summary', ui.node('#studioTool'));
  await ui.context.selectNotebook('B');
  const oldDetail = deferred(); ui.pending.set('/api/notebooks/A', oldDetail);
  const selecting = ui.context.selectNotebook('A');
  ui.streams[0].resolve();
  await Promise.resolve(); await Promise.resolve();
  assert.equal(ui.requests.filter(path => path === '/api/notebooks/A').length, 2); // initial A + pending selection
  ui.pending.delete('/api/notebooks/A');
  ui.notebooks.A = {...ui.notebooks.A, artifacts: [{id: 'artifact-1', title: 'Summary', content: 'Saved', updated_at: '2026-01-01'}]};
  oldDetail.resolve({...ui.notebooks.A, artifacts: []}); // pre-commit snapshot finishes after done
  await selecting;
  await generation;
  assert.equal(ui.requests.filter(path => path === '/api/notebooks/A').length, 3);
  assert.equal(ui.state.detail.artifacts[0].id, 'artifact-1');
});
