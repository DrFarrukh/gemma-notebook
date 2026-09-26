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

test('citation rejection error exposes its reason after delivering the reset event', async () => {
  const stream = mockStream(['{"type":"error","reason":"citation_validation","message":"Answer rejected"}\n',
                             '{"type":"done"}\n']);
  const received = [];
  await assert.rejects(streamRequest('/chat', {}, event => received.push(event)), error =>
    error.reason === 'citation_validation' && error.message === 'Answer rejected');
  assert.deepEqual(received.map(event => event.type), ['error']);
  assert.equal(stream.cancelled, true);
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

test('grouped, adjacent and ranged citations link each known sparse number by metadata index', () => {
  const citations = [7, 3, 12, 1, 10, 2, 5, 4, 6].map(number => ({number}));
  const html = markdown('[3,4] [1,10] [1,2,5,7,10,12] [1][2] [3-5] [5–6] [12, 99]', citations);
  assert.match(html, /\[<button class="citation" data-citation="1" title="Open source 3">3<\/button>, <button class="citation" data-citation="7" title="Open source 4">4<\/button>\]/);
  assert.match(html, /data-citation="4" title="Open source 10">10<\/button>/);
  assert.match(html, /data-citation="2" title="Open source 12">12<\/button>/);
  assert.match(html, /\[<button class="citation" data-citation="2" title="Open source 12">12<\/button>, 99\]/);
  assert.equal((html.match(/data-citation=/g) || []).length, 2 + 2 + 6 + 2 + 3 + 2 + 1);
});

test('C namespace links current citations and leaves academic numeric references plain', () => {
  const citations = [1, 2, 3, 4, 5, 32].map(number => ({number, namespace: 'C'}));
  const html = markdown('Previous work [32], methods [5], [17]; evidence [C1, C3] and [C2-C4].', citations);
  assert.match(html, /Previous work \[32\], methods \[5\], \[17\]/);
  assert.match(html, /Open source C1/);
  assert.match(html, /Open source C3/);
  assert.match(html, /Open source C4/);
  assert.doesNotMatch(html, /Open source C32|Open source C5/);
  assert.equal((html.match(/data-citation=/g) || []).length, 5);
});

test('legacy saved numeric citations remain clickable when no C citations are present', () => {
  const html = markdown('Historical answer [1, 2].', [{number: 1}, {number: 2}]);
  assert.match(html, /Open source 1/);
  assert.match(html, /Open source 2/);
});

test('malformed markers stay escaped, code and numeric Markdown links are never citation buttons', () => {
  const bad = ['[1x]', '[1,]', '[0]', '[1,-2]', '[-1]', '[2-1]', '[1–0]', '[1-201]',
    '[1,2-201]', '[9007199254740992]', '[12345678901234567]', '[1, 2', '[1,,2]'];
  const html = markdown(bad.join(' ') + ' `<b>[1, 2]</b>`\n```\n[1,2]\n```\n[2024 report](url) ![1,2](img.png) [1](url) <img onerror="x">',
    [{number: 1}, {number: 2}]);
  assert.doesNotMatch(html, /data-citation=|<img|<b>/);
  for (const marker of bad) assert.ok(html.includes(marker));
  assert.match(html, /<code>&lt;b&gt;\[1, 2\]&lt;\/b&gt;<\/code>/);
  assert.match(html, /<pre><code>\[1,2\]<\/code><\/pre>/);
  assert.match(html, /&lt;img onerror=&quot;x&quot;&gt;/);
});

test('emphasis spans citation and code tokens but never interprets markup inside code or links', () => {
  const html = markdown('**Claim [1, 2] confirmed** **Uses `<b>**literal**</b>` safely** [**label**](url) **<img>**',
    [{number: 1}, {number: 2}]);
  assert.match(html, /<strong>Claim \[<button[^>]*>1<\/button>, <button[^>]*>2<\/button>\] confirmed<\/strong>/);
  assert.match(html, /<strong>Uses <code>&lt;b&gt;\*\*literal\*\*&lt;\/b&gt;<\/code> safely<\/strong>/);
  assert.match(html, /\[\*\*label\*\*\]\(url\)/);
  assert.match(html, /<strong>&lt;img&gt;<\/strong>/);
  assert.doesNotMatch(html, /<img>|<strong>literal<\/strong>|\[<strong>label/);
});

test('balanced Markdown destinations are opaque; incomplete streaming brackets stay literal', () => {
  const links = '[1,2](https://example.org/Foo_(bar)) ![1-2](https://example.org/plot_(final).png) ' +
    '[1](https://example.org/a\\(b\\)/[2])';
  const html = markdown(links, [{number: 1}, {number: 2}]);
  assert.doesNotMatch(html, /data-citation=/);
  assert.ok(html.includes(links));
  for (const partial of ['Text [12', 'Text [1, 12', 'Text [1-3']) {
    const incomplete = markdown(partial, Array.from({length: 12}, (_, i) => ({number: i + 1})));
    assert.equal(incomplete, `<p>${partial}</p>`);
    assert.doesNotMatch(incomplete, /data-citation=/);
  }
  assert.match(markdown('Text [12]', [{number: 12}]), /data-citation="0"/);
});
