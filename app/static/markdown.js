function escapeHtml(value = '') {
  return value.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function markdown(value, citations = []) {
  const citationIndex = new Map(citations.map((item, index) => [Number(item.number ?? index + 1), index]));
  const inline = text => escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(\d+)\]/g, (_, n) => citationIndex.has(Number(n))
      ? `<button class="citation" data-citation="${citationIndex.get(Number(n))}" title="Open source ${n}">${n}</button>`
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

if (typeof module !== 'undefined' && module.exports) module.exports = {escapeHtml, markdown};
