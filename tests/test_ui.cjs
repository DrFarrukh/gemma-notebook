const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const {markdown, escapeHtml} = require('../app/static/markdown.js');

test('chat page requests fresh UI assets and tolerates a cached previous script', () => {
  const html = readFileSync(require.resolve('../app/static/index.html'), 'utf8');
  assert.match(html, /\/assets\/app\.js\?v=source-status-20260927/);
  assert.match(html, /\/assets\/style\.css\?v=source-status-20260927/);
  assert.match(html, /id="thinkingSelect"/);
  assert.match(html, /id="temperatureSelect"/);
  assert.match(html, /id="chatStatus" class="hidden"/);
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function uiHarness() {
  const nodes = new Map(), requests = [], streams = [], toasts = [];
  const listeners = {};
  function element() {
    const classes = new Set(['hidden']);
    const children = [], actions = [], descendants = new Map();
    return {
      classList: {
        add: key => classes.add(key), remove: key => classes.delete(key),
        toggle(key, on) { if (on === undefined) on = !classes.has(key); if (on) classes.add(key); else classes.delete(key); },
        contains: key => classes.has(key)
      },
      style: {}, scrollHeight: 0, scrollTop: 0, textContent: '', value: '', innerHTML: '', children,
      lastElementChild: {children: actions, append(child) { actions.push(child); }},
      append(child) { children.push(child); if (child.className === 'toast') toasts.push(child.textContent); },
      remove() {}, showModal() { this.open = true; }, setAttribute(key, value) { this[key] = value; }, addEventListener() {},
      querySelector(selector) { if (!descendants.has(selector)) descendants.set(selector, element()); return descendants.get(selector); },
      querySelectorAll() { return []; }
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
    document: {documentElement: {dataset: {}}, querySelector: node, querySelectorAll: () => [], createElement: element,
      addEventListener(name, callback) { listeners[name] = callback; }},
    localStorage: {getItem: () => null, setItem() {}}, setTimeout: () => 0, clearTimeout() {}, setInterval() {},
    AbortController, DOMException, markdown, escapeHtml,
    fetch: async path => {
      requests.push(path);
      if (pending.has(path)) {
        const response = pending.get(path);
        return {ok: true, json: async () => response.promise};
      }
      const body = path === '/api/health' ? {ollama: true, generation_ready: true, embedding_ready: true}
        : path === '/api/settings' ? {generation_model: 'gemma4:e4b', available_models: ['gemma4:e4b'],
            num_ctx: 16384, context_options: [8192, 16384], temperature: 0.2,
            temperature_options: [0, 0.2, 0.7], thinking: 'auto',
            thinking_options: ['auto', 'off', 'low', 'medium', 'high', 'max', 'on']}
        : path === '/api/notebooks' ? Object.values(notebooks)
        : path.endsWith('/messages') ? histories[path.split('/')[3]] : notebooks[path.split('/').at(-1)];
      return {ok: true, json: async () => body};
    },
    streamRequest: (path, body, onEvent, signal) => {
      const gate = deferred();
      const stream = {path, body, onEvent, signal, resolve() { onEvent({type: 'done'}); gate.resolve(); },
        fail(message, reason) {
          onEvent({type: 'error', message, reason});
          const error = new Error(message); error.reason = reason; gate.reject(error);
        }};
      streams.push(stream);
      signal.addEventListener('abort', () => gate.reject(new DOMException('Stopped', 'AbortError')), {once: true});
      return gate.promise;
    }
  });
  vm.runInContext(readFileSync(require.resolve('../app/static/app.js'), 'utf8'), context);
  const state = vm.runInContext('state', context);
  return {context, state, node, notebooks, histories, requests, streams, toasts, pending, listeners, ready: new Promise(setImmediate)};
}

test('generation controls load context, thinking, and temperature settings', async () => {
  const ui = uiHarness(); await ui.ready;
  assert.match(ui.node('#ctxSelect').innerHTML, /value="16384" selected/);
  assert.match(ui.node('#thinkingSelect').innerHTML, /value="auto" selected>Think auto/);
  assert.match(ui.node('#temperatureSelect').innerHTML, /value="0.2" selected>Temp 0.2/);
});

test('source list clearly shows queued, processing, completed, and failed states', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  ui.state.detail.sources = [
    {id:'queued', name:'queued.pdf', status:'queued', enabled:1},
    {id:'processing', name:'processing.pdf', status:'processing', enabled:1},
    {id:'ready', name:'ready.pdf', status:'ready', enabled:1, page_count:8},
    {id:'error', name:'error.pdf', status:'error', enabled:0, error:'OCR failed'}
  ];

  ui.context.renderSources();
  const list = ui.node('#sourceList').innerHTML;
  assert.match(list, /Queued for processing/);
  assert.match(list, /Processing…/);
  assert.match(list, /Processed · 8 pages/);
  assert.match(list, /Not processed · Retry available/);
  assert.match(list, /aria-live="polite"/);
});

test('request meter labels the latest model request rather than chat memory', async () => {
  const ui = uiHarness(); await ui.ready;
  ui.state.numCtx = 16384;
  ui.context.updateContextMeter(6000, 2000);
  const meter = ui.node('#contextMeter');
  assert.equal(meter.textContent, 'Request 49% · 8.0K/16.4K');
  assert.match(meter.title, /most recent model request/);
  assert.match(meter.title, /not cumulative notebook or chat memory/);
  ui.context.updateContextMeter(1000, 1000);
  assert.equal(meter.textContent, 'Request 12% · 2.0K/16.4K');
});

test('citation retry clears the draft and streams into the same assistant bubble', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Make a diagram');
  const stream = ui.streams[0], area = ui.node('#messages');
  const answer = area.children.at(-1), initialCount = area.children.length;
  stream.onEvent({type: 'delta', text: 'Invalid draft (citation 1)'});
  assert.equal(answer._content, 'Invalid draft (citation 1)');
  stream.onEvent({type: 'retry', reason: 'citation_validation', message: 'Retrying with source-citation formatting…'});
  assert.equal(answer._content, '');
  assert.match(answer.querySelector('.message-body').innerHTML, /Retrying with source-citation formatting/);
  stream.onEvent({type: 'delta', text: 'Grounded prose [1].'});
  stream.resolve(); await chat;
  assert.equal(area.children.length, initialCount);
  assert.equal(area.children.at(-1), answer);
  assert.equal(answer._content, 'Grounded prose [1].');
  assert.deepEqual(answer.lastElementChild.children.map(child => child.className),
                   ['answer-metrics', 'message-actions']);
});

test('final citation rejection shows both attempts and adds no answer actions', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Make a diagram');
  const stream = ui.streams[0], answer = ui.node('#messages').children.at(-1);
  stream.onEvent({type: 'delta', text: 'First invalid draft'});
  stream.onEvent({type: 'retry', reason: 'citation_validation', message: 'Retrying with source-citation formatting…'});
  stream.onEvent({type: 'delta', text: 'Second invalid draft'});
  stream.fail('Answer rejected: the model did not provide valid current source citations. Please retry or rephrase the request.', 'citation_validation');
  await chat;
  assert.match(answer._content, /Draft 1\n\nFirst invalid draft/);
  assert.match(answer._content, /Draft 2\n\nSecond invalid draft/);
  assert.match(answer.lastElementChild.children[0].textContent, /Answer rejected: the model did not provide valid current source citations/);
  assert.match(answer.lastElementChild.children[0].textContent, /draft above was not saved/);
  assert.equal(answer.lastElementChild.children.some(child => child.className === 'message-actions'), false);
});

test('an empty retry cannot erase a previous rejected draft', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Compare selected papers');
  const stream = ui.streams[0], answer = ui.node('#messages').children.at(-1);
  const table = '| Paper | Result |\n|---|---|\n| First | Accuracy [1] |';
  stream.onEvent({type: 'delta', text: table});
  stream.onEvent({type: 'retry', reason: 'citation_validation'});
  stream.onEvent({type: 'delta', text: '   '});
  stream.fail('Answer rejected: the model did not provide valid current source citations.', 'citation_validation');
  await chat;
  assert.equal(answer._content, table);
  assert.match(answer.lastElementChild.children[0].textContent, /Answer rejected/);
  assert.equal(answer.lastElementChild.children.some(child => child.className === 'message-actions'), false);
});

test('a rejection without generated text says no draft was returned', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Question');
  const stream = ui.streams[0], answer = ui.node('#messages').children.at(-1);
  stream.fail('Answer rejected: the model did not provide valid current source citations.', 'citation_validation');
  await chat;
  assert.equal(answer._content, 'No answer text was returned.');
  assert.match(answer.lastElementChild.children[0].textContent, /No answer was saved/);
});

test('source completeness rejection keeps the streamed table beside its rejection', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Compare sources');
  const stream = ui.streams[0], answer = ui.node('#messages').children.at(-1);
  const table = '| Source | Result |\n|---|---|\n| Paper A | Result [1] |';
  stream.onEvent({type: 'delta', text: table});
  stream.fail('Answer rejected: the model omitted or duplicated selected sources.', 'source_completeness');
  await chat;
  assert.equal(answer._content, table);
  assert.match(answer.lastElementChild.children[0].textContent, /omitted or duplicated selected sources/);
});

test('progress appears by the typing cursor and completed answers show measured speed and time', async () => {
  const ui = uiHarness(); await ui.ready;
  await ui.context.selectNotebook('A');
  const chat = ui.context.sendQuestion('Question');
  const stream = ui.streams[0], answer = ui.node('#messages').children.at(-1);
  assert.match(answer.querySelector('.message-body').innerHTML, /Finding evidence and generating an answer/);
  stream.onEvent({type: 'status', message: 'Preparing source records…'});
  assert.match(answer.querySelector('.message-body').innerHTML, /Preparing source records/);
  stream.onEvent({type: 'delta', text: 'Grounded [1].'});
  assert.doesNotMatch(answer.querySelector('.message-body').innerHTML, /Preparing source records/);
  stream.onEvent({type: 'done', metrics: {eval_count: 100, eval_duration: 2e9, total_duration: 3e9}});
  stream.resolve(); await chat;
  const metrics = answer.lastElementChild.children.find(child => child.className === 'answer-metrics');
  assert.match(metrics.textContent, /100 output tokens · 50.0 tokens\/s · Model 3.0s/);
  assert.match(metrics.textContent, /Before first token/);
  assert.match(metrics.textContent, /Wall/);
});

test('grouped sparse citations open the correct source, page and excerpt', async () => {
  const ui = uiHarness(); await ui.ready;
  const citations = [
    {number: 3, source_name: 'Third', source_id: 'third', page: 2, excerpt: 'Third excerpt'},
    {number: 10, source_name: 'Tenth', source_id: 'tenth', page: 17, excerpt: 'Tenth excerpt'},
    {number: 1, source_name: 'First', source_id: 'first', page: null, excerpt: 'First excerpt'}
  ];
  const html = markdown('Sources [1, 3, 10]', citations);
  const message = {_citations: citations};
  for (const [number, index] of [[1, 2], [3, 0], [10, 1]]) {
    assert.match(html, new RegExp(`data-citation="${index}" title="Open source ${number}"`));
    const button = {dataset: {citation: String(index)}, closest(selector) {
      return selector === '[data-citation]' ? this : selector === '.message' ? message : null;
    }};
    await ui.listeners.click({target: button});
    const item = citations[index];
    assert.equal(ui.node('#citationTitle').textContent, item.source_name);
    assert.equal(ui.node('#citationLocation').textContent, item.page ? `Page ${item.page}` : 'Source excerpt');
    assert.equal(ui.node('#citationExcerpt').textContent, item.excerpt);
    assert.equal(ui.node('#citationFile').href, `/api/files/${item.source_id}${item.page ? `#page=${item.page}` : ''}`);
    assert.equal(ui.node('#citationDialog').open, true);
  }
});

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
  assert.equal(ui.state.run.status, 'Finding evidence and generating an answer…');
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
  assert.match(ui.toasts.at(-1), /Generation interrupted in Notebook A; refresh to view history/);
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
