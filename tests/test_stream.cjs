const {test} = require('node:test');
const assert = require('node:assert/strict');
const {streamRequest} = require('../app/static/stream.js');
const {markdown} = require('../app/static/markdown.js');

function mockStream(chunks, {status = 200, signal} = {}) {
  const bytes = chunks.map(chunk => typeof chunk === 'string' ? new TextEncoder().encode(chunk) : chunk);
  let cancelled = false, lockedDuringRead = false;
  const stream = new ReadableStream({
    pull(controller) {
      lockedDuringRead = stream.locked;
      if (bytes.length) controller.enqueue(bytes.shift());
      else controller.close();
    },
    cancel() { cancelled = true; }
  });
  global.fetch = async (_path, options) => {
    assert.equal(options.signal, signal);
    return {ok: status === 200, status, statusText: 'Bad Request', body: stream};
  };
  return {stream, get cancelled() { return cancelled; }, get lockedDuringRead() { return lockedDuringRead; }};
}

test('split NDJSON, final un-terminated done, multibyte UTF-8 and unknown events', async () => {
  const utf8 = new TextEncoder().encode('😀');
  const stream = mockStream([
    '{"type":"cit',
    'ations","citations":[]}\n{"type":"delta","text":"',
    utf8.slice(0, 2), utf8.slice(2),
    '"}\n{"type":"status","message":"Working"}\n{"type":"done"}'
  ]);
  const events = [];
  await streamRequest('/chat', {}, event => events.push(event));
  assert.deepEqual(events.map(event => event.type), ['citations', 'delta', 'status', 'done']);
  assert.equal(events[1].text, '😀');
  assert.equal(stream.stream.locked, false);
  assert.equal(stream.lockedDuringRead, true);
});

test('missing done rejects partial output and releases the reader', async () => {
  const stream = mockStream(['{"type":"delta","text":"partial"}\n']);
  const events = [];
  await assert.rejects(streamRequest('/chat', {}, event => events.push(event)), /before generation completed/);
  assert.equal(events[0].text, 'partial');
  assert.equal(stream.stream.locked, false);
});

test('upstream error and malformed JSON reject and cancel the reader', async () => {
  for (const [line, expected] of [
    ['{"type":"error","message":"offline"}\n', /offline/],
    ['{"type": }\n', /Malformed streaming response/],
    ['null\n', /Malformed streaming response/]
  ]) {
    const stream = mockStream([line, '{"type":"done"}\n']);
    await assert.rejects(streamRequest('/chat', {}, () => {}), expected);
    assert.equal(stream.cancelled, true);
    assert.equal(stream.stream.locked, false);
  }
});

test('aborting while processing cancels the reader and rejects', async () => {
  const controller = new AbortController();
  const stream = mockStream(['{"type":"delta","text":"partial"}\n', '{"type":"done"}\n'], {signal: controller.signal});
  await assert.rejects(streamRequest('/chat', {}, () => controller.abort(), controller.signal), {name: 'AbortError'});
  assert.equal(stream.cancelled, true);
  assert.equal(stream.stream.locked, false);
});

test('missing body is a protocol error', async () => {
  global.fetch = async () => ({ok: true, body: null});
  await assert.rejects(streamRequest('/chat', {}, () => {}), /no body/);
});

test('decoder flush detects an incomplete UTF-8 event before done', async () => {
  const stream = mockStream([new Uint8Array([0xf0, 0x9f])]);
  await assert.rejects(streamRequest('/chat', {}, () => {}), /Malformed streaming response/);
  assert.equal(stream.stream.locked, false);
});

test('HTTP error is reported without attempting a reader', async () => {
  global.fetch = async () => ({ok: false, status: 503, statusText: 'Unavailable', json: async () => ({detail: 'Try again'})});
  await assert.rejects(streamRequest('/chat', {}, () => {}), /Try again/);
});

test('successful completion also cancels and releases its reader', async () => {
  let cancelled = false, released = false, reads = 0;
  global.fetch = async () => ({ok: true, body: {getReader() {
    return {
      async read() { reads++; if (reads > 1) return new Promise(() => {}); return {value: new TextEncoder().encode('{"type":"done"}\n{"type":"error","message":"late"}\n'), done: false}; },
      async cancel() { cancelled = true; },
      releaseLock() { released = true; }
    };
  }}});
  await streamRequest('/chat', {}, () => {});
  assert.equal(cancelled, true);
  assert.equal(released, true);
  assert.equal(reads, 1);
});

test('done is success even if the caller aborts synchronously from its callback', async () => {
  const controller = new AbortController();
  const stream = mockStream(['{"type":"do', 'ne"}'], {signal: controller.signal});
  await streamRequest('/chat', {}, event => { if (event.type === 'done') controller.abort(); }, controller.signal);
  assert.equal(controller.signal.aborted, true);
  assert.equal(stream.stream.locked, false);
});

test('sparse explicit citation numbers resolve by array position and escape model content', () => {
  const html = markdown('<img src=x onerror=alert(1)> [7] [1] &', [
    {number: 2, source_name: '<unsafe>'}, {number: 7, source_name: 'safe'}
  ]);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html, /data-citation="1" title="Open source 7">7<\/button>/);
  assert.match(html, /\[1\]/);
  assert.match(html, /&amp;<\/p>/);
  assert.doesNotMatch(html, /<img|<unsafe>/);
  assert.match(markdown('`<script>`'), /<code>&lt;script&gt;<\/code>/);
});
