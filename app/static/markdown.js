function escapeHtml(value = '') {
  return value.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

// Keep this grammar and limits in sync with chat.references().
function citationNumbers(inner) {
  const term = '[0-9]+(?:[ \\t]*[-–][ \\t]*[0-9]+)?';
  if (!new RegExp(`^${term}(?:[ \\t]*,[ \\t]*${term})*$`).test(inner)) return null;
  const numbers = [];
  const parse = value => value.length <= 16 && Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : null;
  for (const part of inner.split(/[ \t]*,[ \t]*/)) {
    const ends = part.split(/[ \t]*[-–][ \t]*/).map(parse);
    if (ends.includes(null) || (ends.length === 2 && ends[1] < ends[0])) return null;
    const count = ends.length === 2 ? ends[1] - ends[0] + 1 : 1;
    if (count > 200 - numbers.length) return null;
    for (let n = ends[0]; n < ends[0] + count; n++) numbers.push(n);
  }
  return numbers;
}

// Return the end of a same-line Markdown destination, or null. Match chat.py's bound.
function linkDestinationEnd(text, start) {
  if (text[start] !== '(') return null;
  let depth = 0;
  for (let i = start; i < text.length && i - start <= 2048; i++) {
    if (text[i] === '\\') {
      if (i + 1 >= text.length || i + 1 - start > 2048 || text[i + 1] === '\n') return null;
      i++; continue;
    }
    if (text[i] === '\n') return null;
    if (text[i] === '(' && ++depth > 32) return null;
    if (text[i] === ')' && --depth === 0) return i + 1;
  }
  return null;
}

function markdown(value, citations = []) {
  const citationIndex = new Map(citations.map((item, index) => [Number(item.number ?? index + 1), index]));
  const inline = text => {
    const parts = [];
    const addPlain = value => {
      for (const piece of value.split(/(\*\*)/)) {
        if (piece) parts.push(piece === '**' ? {bold: true} : {html: escapeHtml(piece)});
      }
    };
    let end = 0;
    const tokens = /`[^`]+`|\[[^\]\n]*(?:\]|(?=\n|$))/g;
    for (const match of text.matchAll(tokens)) {
      if (match.index < end) continue;
      addPlain(text.slice(end, match.index));
      const raw = match[0];
      if (raw.startsWith('`')) {
        parts.push({html: `<code>${escapeHtml(raw.slice(1, -1))}</code>`});
      } else {
        const inner = raw.slice(1, -1);
        const linkEnd = raw.endsWith(']') ? linkDestinationEnd(text, match.index + raw.length) : null;
        const numbers = raw.endsWith(']') && !linkEnd && /^[0-9]/.test(inner) ? citationNumbers(inner) : null;
        if (linkEnd !== null) {
          parts.push({html: escapeHtml(text.slice(match.index, linkEnd))});
          end = linkEnd;
          continue;
        } else if (numbers) {
          parts.push({html: `[${numbers.map(n => citationIndex.has(n)
            ? `<button class="citation" data-citation="${citationIndex.get(n)}" title="Open source ${n}">${n}</button>`
            : String(n)).join(', ')}]`});
        } else addPlain(raw);
      }
      end = match.index + raw.length;
    }
    addPlain(text.slice(end));
    // Pair emphasis only outside opaque code and Markdown links; citation HTML stays inside it.
    const open = [];
    for (let i = 0; i < parts.length; i++) {
      if (!parts[i].bold) continue;
      if (open.length && i > open.at(-1) + 1) {
        parts[open.pop()].html = '<strong>';
        parts[i].html = '</strong>';
      } else open.push(i);
    }
    return parts.map(part => part.html ?? '**').join('');
  };
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

if (typeof module !== 'undefined' && module.exports) module.exports = {escapeHtml, markdown};
