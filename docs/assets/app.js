/* 若璩 Ruoqu showcase — vanilla JS + d3 v7 */
(() => {
'use strict';
const T0 = -800, T1 = 1900;
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
// language: ?lang=en|zhs|zht in the URL wins, then the saved choice, then simplified Chinese
const QLANG = new URLSearchParams(location.search).get('lang');
let LANG = (['zhs', 'zht', 'en'].includes(QLANG) && QLANG) || store.get('ruoqu-lang2') || 'zhs';   // zhs 简体 (default) · zht 繁體 · en
if (!['zhs', 'zht', 'en'].includes(LANG)) LANG = 'zhs';
const isZh = () => LANG !== 'en';
/* traditional -> simplified, character by character (maps built with OpenCC, assets/zhconv.js) */
const PHRASE = [['带著', '带着'], ['看著', '看着'], ['接著', '接着'], ['跟著', '跟着'], ['随著', '随着'], ['著重', '着重'], ['意味著', '意味着']];
const t2s = (str) => { if (str == null) return str; let o = ''; for (const c of String(str)) o += (window.T2S && T2S[c]) || c;
  for (const [a, b] of PHRASE) o = o.split(a).join(b); return o; };
const tx = (str) => LANG === 'zhs' ? t2s(str) : str;
const wrapUI = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === 'function' ? (...a) => t2s(v(...a)) : (typeof v === 'object' ? wrapUI(v) : t2s(v))]));
let UIS = null;
const ui = () => LANG === 'en' ? UI.en : LANG === 'zht' ? UI.zh : (UIS || (UIS = wrapUI(UI.zh)));
const dynName = (d) => isZh() ? tx(d.zh) : d.en;
/* rough label width, to show a dynasty name only where it fits */
const fits = (d, px) => px > (isZh() ? tx(d.zh).length * 13 : d.en.length * 6.6) + 6;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const DATA = {};
const renderers = [];   // re-run on language/theme change or resize

function fy(y, long) {
  if (y == null) return '—';
  y = Math.round(y);
  if (isZh()) return y < 0 ? `前${-y}` : `${y}`;
  return y < 0 ? `${-y} BCE` : (long ? `${y} CE` : `${y}`);
}
function rangeTxt(a, b) { return a === b ? fy(a, true) : `${fy(a)}–${fy(b, true)}`; }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function parseRange(s) {
  const m = String(s || '').match(/(-?\d+)\s*[~～–-]\s*(-?\d+)/);
  if (m) return [+m[1], +m[2]];
  const k = String(s || '').match(/-?\d+/); return k ? [+k[0], +k[0]] : null;
}

/* ---------- tooltip ---------- */
const tip = document.getElementById('tip');
function showTip(ev, rows) {   // rows: [{v, l, c}] value leads, label follows; strings via textContent
  tip.replaceChildren();
  rows.forEach((r, i) => {
    const d = el('div');
    if (r.c) { const k = el('span', 'tk'); k.style.borderColor = r.c; d.appendChild(k); }
    if (r.v != null) d.appendChild(el('span', i === 0 && !r.c ? 'tv' : 'tv', r.v));
    if (r.l) { d.appendChild(document.createTextNode(' ')); d.appendChild(el('span', 'tl', r.l)); }
    tip.appendChild(d);
  });
  tip.classList.add('on'); moveTip(ev);
}
function moveTip(ev) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
function hideTip() { tip.classList.remove('on'); }

/* ---------- shared time axis with dynasty bands ---------- */
function timeScale(w, m) { return d3.scaleLinear().domain([T0, T1]).range([m.l, w - m.r]); }
function dynBands(g, x, y0, y1, labels = true, labelY = null) {
  const b = g.append('g').attr('class', 'dyn');
  DYN.forEach((d, i) => {
    const a = Math.max(T0, d.a), c = Math.min(T1, d.b);
    if (i % 2 === 0) b.append('rect').attr('class', 'dyn-band').attr('x', x(a)).attr('width', Math.max(0, x(c) - x(a))).attr('y', y0).attr('height', y1 - y0);
    if (labels && fits(d, x(c) - x(a))) b.append('text').attr('class', 'dyn-lbl').attr('x', (x(a) + x(c)) / 2).attr('y', labelY ?? (y0 + 14))
      .attr('text-anchor', 'middle').text(dynName(d));
  });
  return b;
}
function timeAxis(g, x, y, ticks) {
  let tv = ticks || [-800, -500, -200, 100, 400, 700, 1000, 1300, 1600, 1900];
  const minGap = isZh() ? 44 : 58;   // keep tick labels apart on narrow screens
  tv = tv.filter((t, i, a) => i === 0 || Math.abs(x(t) - x(a[0])) >= minGap * i ? true : false);
  for (let step = 1; step < tv.length && tv.some((t, i) => i && x(t) - x(tv[i - 1]) < minGap); step++) tv = tv.filter((t, i) => i % 2 === 0);
  const ax = d3.axisBottom(x).tickValues(tv).tickFormat(d => fy(d)).tickSize(4);
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${y})`).call(ax).call(s => s.select('.domain').attr('stroke', css('--rule')));
}
function svgIn(node, w, h) {
  node.replaceChildren();
  return d3.select(node).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('preserveAspectRatio', 'xMidYMid meet');
}
const W = (node) => Math.max(320, Math.round(node.getBoundingClientRect().width || 800));

/* posterior curve on the 25-year grid from -800 (per-mille) */
function curvePts(item) { return item.curve.map((v, i) => ({ t: -800 + 25 * i, v })); }

/* ---------- i18n ---------- */
const ZH = {};
document.querySelectorAll('[data-i18n]').forEach(n => { ZH[n.getAttribute('data-i18n')] = n.textContent; });
document.querySelectorAll('[data-i18n-ph]').forEach(n => { ZH['ph:' + n.getAttribute('data-i18n-ph')] = n.getAttribute('placeholder'); });
function applyLang() {
  document.documentElement.lang = LANG === 'zhs' ? 'zh-Hans' : LANG === 'zht' ? 'zh-Hant' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(n => { const k = n.getAttribute('data-i18n'); n.textContent = isZh() ? tx(ZH[k] ?? '') : (EN[k] ?? ZH[k]); });
  document.querySelectorAll('[data-i18n-ph]').forEach(n => { const k = n.getAttribute('data-i18n-ph'); n.placeholder = isZh() ? tx(ZH['ph:' + k]) : (EN[k] ?? ''); });
  document.querySelectorAll('[data-num]').forEach(n => { const k = +n.getAttribute('data-num'); n.textContent = isZh() ? NUMS[LANG][k - 1] : String(k); });
  document.querySelectorAll('.seg-b').forEach(b => b.setAttribute('aria-pressed', b.getAttribute('data-lang') === LANG));
  document.title = LANG === 'en' ? '若璩 (Ruoqu) · A language clock for Classical Chinese' : tx('若璩 · 語言之鐘');
}
document.querySelectorAll('.seg-b').forEach(b => b.addEventListener('click', () => { LANG = b.getAttribute('data-lang'); store.set('ruoqu-lang2', LANG);
  try { const u = new URL(location.href); u.searchParams.set('lang', LANG); history.replaceState(null, '', u); } catch (e) {}
  applyLang(); rerender(); }));
(() => {
  const t = store.get('ruoqu-theme'); if (t) document.documentElement.dataset.theme = t;
  document.getElementById('theme').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'light' : 'dark'; store.set('ruoqu-theme', document.documentElement.dataset.theme); rerender();
  });
})();
function rerender() { renderers.forEach(f => { try { f(); } catch (e) { console.error(e); } }); }
let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(rerender, 150); });

/* ================= HERO DEMO ================= */
function hero() {
  const items = DATA.quiz.items.filter(d => ['KR3l0002', 'KR3j0092', 'KR3j0080', 'KR3a0047'].includes(d.id));
  let k = 0, timer = null;
  const textBox = document.getElementById('demo-text'), cap = document.getElementById('demo-caption'), box = document.getElementById('demo-chart');
  function draw(item, animate) {
    const w = W(box), h = 140, m = { l: 10, r: 10, t: 10, b: 24 };
    const svg = svgIn(box, w, h), x = timeScale(w, m);
    const y = d3.scaleLinear().domain([0, d3.max(item.curve) * 1.08]).range([h - m.b, m.t]);
    dynBands(svg, x, m.t, h - m.b, true, h - m.b - 6);
    timeAxis(svg, x, h - m.b);
    const pts = curvePts(item);
    const area = d3.area().x(d => x(d.t)).y0(h - m.b).y1(d => y(d.v)).curve(d3.curveMonotoneX);
    const line = d3.line().x(d => x(d.t)).y(d => y(d.v)).curve(d3.curveMonotoneX);
    svg.append('rect').attr('x', x(item.lo)).attr('width', Math.max(2, x(item.hi) - x(item.lo))).attr('y', m.t).attr('height', h - m.b - m.t)
      .attr('fill', css('--scholar')).attr('opacity', .16);
    const a = svg.append('path').attr('d', area(pts)).attr('fill', css('--model')).attr('opacity', .14);
    const l = svg.append('path').attr('d', line(pts)).attr('fill', 'none').attr('stroke', css('--model')).attr('stroke-width', 2);
    const mk = svg.append('g').attr('transform', `translate(${x(item.median)},0)`);
    mk.append('line').attr('y1', m.t).attr('y2', h - m.b).attr('stroke', css('--model')).attr('stroke-dasharray', '3 3');
    if (animate && !reduce) {
      const L = l.node().getTotalLength();
      l.attr('stroke-dasharray', `${L} ${L}`).attr('stroke-dashoffset', L).transition().delay(1200).duration(1600).attr('stroke-dashoffset', 0);
      a.attr('opacity', 0).transition().delay(2200).duration(800).attr('opacity', .14);
      mk.attr('opacity', 0).transition().delay(2800).duration(400).attr('opacity', 1);
    }
  }
  function show(animate) {
    const item = items[k % items.length];
    textBox.replaceChildren();
    [...tx(item.text).slice(0, 44)].forEach((c, i) => {
      const s = el('span', 'ch', c); textBox.appendChild(s);
      if (animate && !reduce) setTimeout(() => s.classList.add('on'), 25 * i); else s.classList.add('on');
    });
    draw(item, animate);
    const t = isZh() ? tx(item.title) : item.en;
    cap.innerHTML = '';
    setTimeout(() => { cap.innerHTML = ui().demo(t.replace(/[<>&]/g, ''), fy(item.median, true)); }, animate && !reduce ? 3000 : 0);
  }
  function loop() { show(true); timer = setTimeout(() => { k++; loop(); }, 8000); }
  renderers.push(() => { clearTimeout(timer); show(false); timer = setTimeout(() => { k++; loop(); }, 8000); });
  loop();
}

/* ================= QUIZ ================= */
function quiz() {
  const items = d3.shuffle(DATA.quiz.items.slice());
  let k = 0, guess = null, done = 0, sumU = 0, sumM = 0;
  const box = document.getElementById('quiz-chart'), txt = document.getElementById('quiz-text'), meta = document.getElementById('quiz-meta'),
        res = document.getElementById('quiz-result'), sc = document.getElementById('quiz-score');
  const dist = (y, a, b) => Math.max(0, a - y, y - b);
  function draw() {
    const item = items[k % items.length];
    txt.textContent = tx(item.text);
    meta.textContent = guess == null ? ui().quiz_click : '';
    if (guess != null) {
      const b = el('b', null, isZh() ? `《${tx(item.title)}》` : item.en);
      meta.append(ui().quiz_from + ' ', b, `  ·  ${ui().quiz_written} ${rangeTxt(item.lo, item.hi)}`);
    }
    const w = W(box), h = 220, m = { l: 12, r: 12, t: 14, b: 26 };
    const svg = svgIn(box, w, h), x = timeScale(w, m);
    dynBands(svg, x, m.t, h - m.b, true, m.t + 14);
    timeAxis(svg, x, h - m.b);
    const hit = svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', w - m.l - m.r).attr('height', h - m.t - m.b)
      .attr('fill', 'transparent').style('cursor', guess == null ? 'crosshair' : 'default');
    const hover = svg.append('line').attr('y1', m.t).attr('y2', h - m.b).attr('stroke', css('--ink-3')).attr('stroke-dasharray', '2 3').attr('opacity', 0);
    if (guess == null) {
      hit.on('pointermove', (ev) => { const [px] = d3.pointer(ev); hover.attr('x1', px).attr('x2', px).attr('opacity', 1); showTip(ev, [{ v: fy(Math.round(x.invert(px) / 25) * 25, true) }]); })
         .on('pointerleave', () => { hover.attr('opacity', 0); hideTip(); })
         .on('click', (ev) => { const [px] = d3.pointer(ev); guess = Math.round(x.invert(px) / 25) * 25; hideTip(); reveal(item); draw(); });
      res.textContent = '';
      return;
    }
    const y = d3.scaleLinear().domain([0, d3.max(item.curve) * 1.1]).range([h - m.b, m.t + 24]);
    svg.append('rect').attr('x', x(item.lo)).attr('width', Math.max(3, x(item.hi) - x(item.lo))).attr('y', m.t + 20).attr('height', h - m.b - m.t - 20)
      .attr('fill', css('--scholar')).attr('opacity', .18);
    const pts = curvePts(item);
    svg.append('path').attr('d', d3.area().x(d => x(d.t)).y0(h - m.b).y1(d => y(d.v)).curve(d3.curveMonotoneX)(pts)).attr('fill', css('--model')).attr('opacity', .13);
    const l = svg.append('path').attr('d', d3.line().x(d => x(d.t)).y(d => y(d.v)).curve(d3.curveMonotoneX)(pts)).attr('fill', 'none').attr('stroke', css('--model')).attr('stroke-width', 2);
    if (!reduce) { const L = l.node().getTotalLength(); l.attr('stroke-dasharray', `${L} ${L}`).attr('stroke-dashoffset', L).transition().duration(900).attr('stroke-dashoffset', 0); }
    const mark = (t, color, label, dy) => {
      const g = svg.append('g').attr('transform', `translate(${x(t)},0)`);
      g.append('line').attr('y1', m.t + 20).attr('y2', h - m.b).attr('stroke', color).attr('stroke-width', 2);
      g.append('text').attr('y', m.t + 32 + dy).attr('x', 5).attr('class', 'lbl').style('fill', css('--ink-2')).text(`${label} ${fy(t)}`);
    };
    mark(guess, css('--ink'), ui().you, 0);
    mark(item.median, css('--model'), ui().model, 16);
    const legend = [{ c: css('--scholar'), t: ui().truth }, { c: css('--model'), t: ui().model }, { c: css('--ink'), t: ui().you }];
    const lg = svg.append('g').attr('transform', `translate(${w - m.r - 300},${h - m.b - 12})`);
    legend.forEach((d, i) => { lg.append('line').attr('x1', i * 100).attr('x2', i * 100 + 14).attr('stroke', d.c).attr('stroke-width', 3);
      lg.append('text').attr('x', i * 100 + 18).attr('y', 4).attr('class', 'lbl').text(d.t); });
  }
  function reveal(item) {
    const u = dist(guess, item.lo, item.hi), mm = dist(item.median, item.lo, item.hi);
    done++; sumU += u; sumM += mm;
    res.innerHTML = `<div class="big">${ui().quiz_err(Math.round(u), Math.round(mm))}</div>` + (item.id === 'KR1c0001' ? `<div>${ui().canon_note}</div>` : '');
    sc.textContent = ui().quiz_score(Math.round(sumU / done), Math.round(sumM / done), done);
  }
  document.getElementById('quiz-next').addEventListener('click', () => { k++; guess = null; draw(); });
  renderers.push(() => { draw(); if (guess != null) reveal2(); });
  function reveal2() { const item = items[k % items.length]; const u = dist(guess, item.lo, item.hi), mm = dist(item.median, item.lo, item.hi);
    res.innerHTML = `<div class="big">${ui().quiz_err(Math.round(u), Math.round(mm))}</div>` + (item.id === 'KR1c0001' ? `<div>${ui().canon_note}</div>` : '');
    if (done) sc.textContent = ui().quiz_score(Math.round(sumU / done), Math.round(sumM / done), done); }
  draw();
}

/* ================= WORDS ================= */
function words() {
  const L = DATA.life, E = DATA.eras, yrs = L.years;
  const colors = () => [css('--model'), css('--scholar'), css('--third')];
  let sel = ['之', '的'];
  const box = document.getElementById('word-chart'), legend = document.getElementById('word-legend'), input = document.getElementById('word-input');
  const series = () => sel.filter(w => L.items[w]).map(w => ({ w, n: L.items[w][0], v: L.items[w].slice(1).map(q => Math.exp(q * L.scale)) }));
  function draw() {
    const S = series(), C = colors();
    const w = W(box), h = 340, m = { l: 52, r: 16, t: 28, b: 28 };
    const svg = svgIn(box, w, h);
    const x = d3.scaleLinear().domain([-600, 1900]).range([m.l, w - m.r]);
    const ext = d3.extent(S.flatMap(s => s.v).concat([1 / 1.25, 1.25]));
    const y = d3.scaleLog().domain([Math.min(ext[0], 1 / ext[1]), Math.max(ext[1], 1 / ext[0])]).range([h - m.b, m.t]).nice();
    DYN.forEach((d, i) => { const a = Math.max(-600, d.a), c = Math.min(T1, d.b); if (c <= a) return;
      if (i % 2 === 0) svg.append('rect').attr('class', 'dyn-band').attr('x', x(a)).attr('width', x(c) - x(a)).attr('y', m.t).attr('height', h - m.b - m.t);
      if (fits(d, x(c) - x(a))) svg.append('text').attr('class', 'dyn-lbl').attr('x', (x(a) + x(c)) / 2).attr('y', m.t - 8).attr('text-anchor', 'middle').text(dynName(d)); });
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickValues([-600, -300, 0, 300, 600, 900, 1200, 1500, 1800]).tickFormat(d => fy(d)).tickSize(4));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, '~g').tickFormat(d => `×${d3.format('~g')(d)}`).tickSize(-(w - m.l - m.r)))
      .call(g => g.selectAll('.tick line').attr('stroke', css('--rule')).attr('stroke-dasharray', '2 3')).call(g => g.select('.domain').remove());
    svg.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(1)).attr('y2', y(1)).attr('stroke', css('--ink-3')).attr('stroke-width', 1);
    svg.append('text').attr('class', 'lbl').attr('x', m.l + 4).attr('y', y(1) - 5).text(isZh() ? '平均' : 'average');
    const line = d3.line().x((d, i) => x(yrs[i])).y(d => y(d)).curve(d3.curveMonotoneX);
    S.forEach((s, i) => {
      svg.append('path').attr('d', line(s.v)).attr('fill', 'none').attr('stroke', C[i]).attr('stroke-width', 2.2);
      const j = d3.maxIndex(s.v);
      svg.append('text').attr('x', x(yrs[j])).attr('y', y(s.v[j]) - 8).attr('text-anchor', 'middle').style('font-family', 'var(--serif-zh)').style('font-size', '18px')
        .style('fill', css('--ink')).text(tx(s.w));
    });
    const cross = svg.append('line').attr('y1', m.t).attr('y2', h - m.b).attr('stroke', css('--ink-3')).attr('opacity', 0);
    const dots = svg.append('g');
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', w - m.l - m.r).attr('height', h - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px] = d3.pointer(ev); const i = d3.minIndex(yrs, t => Math.abs(x(t) - px));
        cross.attr('x1', x(yrs[i])).attr('x2', x(yrs[i])).attr('opacity', 1);
        dots.selectAll('circle').data(S).join('circle').attr('r', 4).attr('cx', x(yrs[i])).attr('cy', s => y(s.v[i])).attr('fill', (s, k) => C[k]).attr('stroke', css('--paper')).attr('stroke-width', 2);
        showTip(ev, [{ v: `${fy(yrs[i])}–${fy(yrs[i] + 99)}` }].concat(S.map((s, k) => ({ c: C[k], v: `×${s.v[i].toFixed(2)}`, l: tx(s.w) }))));
      })
      .on('pointerleave', () => { cross.attr('opacity', 0); dots.selectAll('circle').remove(); hideTip(); });
    legend.replaceChildren();
    S.forEach((s, i) => { const k = el('span', 'key'); const sw = el('span', 'sw'); sw.style.borderColor = C[i]; k.append(sw, `${tx(s.w)}${LANG === 'zhs' && tx(s.w) !== s.w ? '（' + s.w + '）' : ''}  (n=${s.n.toLocaleString()})`); legend.appendChild(k); });
    const miss = sel.filter(w => !L.items[w]);
    if (miss.length) legend.appendChild(el('span', 'hint', `${miss.join('、')}: ${ui().notfound}`));
  }
  const toTrad = (w) => L.items[w] ? w : [...w].map(c => L.items[c] || !window.S2T ? c : (S2T[c] || c)).join('');
  function add(wd) { wd = toTrad((wd || '').trim()); if (!wd) return; sel = sel.filter(s => s !== wd); sel.push(wd); if (sel.length > 3) sel = sel.slice(-3); draw(); }
  document.getElementById('word-add').addEventListener('click', () => { add(input.value); input.value = ''; });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { add(input.value); input.value = ''; } });
  document.getElementById('word-clear').addEventListener('click', () => { sel = []; draw(); });
  const chips = document.getElementById('pair-chips');
  function drawChips() { chips.replaceChildren(); E.pairs.forEach(p => { const b = el('button', 'chip', tx(p.group)); b.addEventListener('click', () => { sel = p.lines.map(l => l.item).slice(0, 3); draw(); document.getElementById('word-chart').scrollIntoView({ block: 'nearest' }); }); chips.appendChild(b); }); }
  drawChips();
  const strip = document.getElementById('era-strip');
  function drawStrip() {
    strip.replaceChildren();
    E.eras.forEach(e => {
      const col = el('div', 'era-col'); col.appendChild(el('div', 'era-y', fy(e.year)));
      const ws = el('div', 'era-w');
      e.words.filter(w => L.items[w]).slice(0, 8).forEach(w => { const b = el('button', null, tx(w)); b.addEventListener('click', () => { sel = [w]; draw(); box.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' }); }); ws.appendChild(b); });
      col.appendChild(ws); strip.appendChild(col);
    });
  }
  renderers.push(() => { draw(); drawStrip(); drawChips(); });
  draw(); drawStrip();
}

/* ================= SCAN ================= */
function scatter() {
  const S = DATA.scatter.map(d => ({ ...d, x: (d.sch[0] + d.sch[1]) / 2, c: (d.cat[0] + d.cat[1]) / 2, mm: d.model[1] }));
  const box = document.getElementById('scatter');
  function draw() {
    const w = W(box), h = Math.round(w / 1.15), m = { l: 58, r: 14, t: 12, b: 40 };
    const svg = svgIn(box, w, h);
    const dom = [-1700, 1950];
    const x = d3.scaleLinear().domain(dom).range([m.l, w - m.r]), y = d3.scaleLinear().domain(dom).range([h - m.b, m.t]);
    const ticks = [-1500, -1000, -500, 0, 500, 1000, 1500];
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickValues(ticks).tickFormat(d => fy(d)).tickSize(-(h - m.t - m.b)))
      .call(g => g.selectAll('.tick line').attr('stroke', css('--rule')).attr('stroke-dasharray', '2 3'));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(d => fy(d)).tickSize(-(w - m.l - m.r)))
      .call(g => g.selectAll('.tick line').attr('stroke', css('--rule')).attr('stroke-dasharray', '2 3'));
    svg.append('text').attr('class', 'lbl').attr('x', (m.l + w - m.r) / 2).attr('y', h - 6).attr('text-anchor', 'middle').text(ui().scholarly);
    svg.append('line').attr('x1', x(dom[0])).attr('y1', y(dom[0])).attr('x2', x(dom[1])).attr('y2', y(dom[1])).attr('stroke', css('--ink-3')).attr('stroke-width', 1);
    const g = svg.append('g');
    g.selectAll('line.link').data(S).join('line').attr('x1', d => x(d.x)).attr('x2', d => x(d.x)).attr('y1', d => y(d.c)).attr('y2', d => y(d.mm))
      .attr('stroke', css('--neutral')).attr('stroke-opacity', .45).attr('stroke-width', 1);
    g.selectAll('circle.cat').data(S).join('circle').attr('cx', d => x(d.x)).attr('cy', d => y(d.c)).attr('r', 4.5).attr('fill', 'none').attr('stroke', css('--neutral')).attr('stroke-width', 1.5);
    g.selectAll('circle.mod').data(S).join('circle').attr('cx', d => x(d.x)).attr('cy', d => y(d.mm)).attr('r', 4.5).attr('fill', css('--model')).attr('stroke', css('--paper')).attr('stroke-width', 1.5);
    const hl = svg.append('circle').attr('r', 9).attr('fill', 'none').attr('stroke', css('--ink')).attr('stroke-width', 1.5).attr('opacity', 0);
    const pts = S.flatMap(d => [[x(d.x), y(d.mm), d], [x(d.x), y(d.c), d]]);
    const del = d3.Delaunay.from(pts, p => p[0], p => p[1]);
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', w - m.l - m.r).attr('height', h - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px, py] = d3.pointer(ev); const i = del.find(px, py); const p = pts[i];
        if (Math.hypot(p[0] - px, p[1] - py) > 30) { hl.attr('opacity', 0); hideTip(); return; }
        const d = p[2]; hl.attr('cx', x(d.x)).attr('cy', y(d.mm)).attr('opacity', 1);
        showTip(ev, [{ v: tx(d.title) }, { l: ui().sets[d.set] || d.set },
          { c: css('--scholar'), v: rangeTxt(d.sch[0], d.sch[1]), l: ui().scholarly },
          { c: css('--neutral'), v: rangeTxt(d.cat[0], d.cat[1]), l: ui().catalogue },
          { c: css('--model'), v: fy(d.mm, true), l: `${ui().model} (${ui().interval} ${fy(d.model[0])}–${fy(d.model[2])})` }]);
      }).on('pointerleave', () => { hl.attr('opacity', 0); hideTip(); });
    const lg = document.getElementById('scatter-legend'); lg.replaceChildren();
    [[css('--model'), ui().model, true], [css('--neutral'), ui().catalogue, false]].forEach(([c, t, f]) => {
      const k = el('span', 'key'), d = el('span', 'dot'); d.style.background = f ? c : 'transparent'; d.style.border = `1.5px solid ${c}`; k.append(d, t); lg.appendChild(k); });
    const tb = document.getElementById('scatter-table'); tb.replaceChildren();
    const t = el('table'), hd = el('tr');
    ['', ui().scholarly, ui().catalogue, ui().model].forEach(s => hd.appendChild(el('th', null, s))); t.appendChild(hd);
    S.slice().sort((a, b) => a.x - b.x).forEach(d => { const r = el('tr'); [tx(d.title), rangeTxt(d.sch[0], d.sch[1]), rangeTxt(d.cat[0], d.cat[1]), fy(d.mm)].forEach(s => r.appendChild(el('td', null, s))); t.appendChild(r); });
    tb.appendChild(t);
  }
  renderers.push(draw); draw();
}

function miniBar(svgNode, segs) {  // segs: [{a,b,c,w,dot}]
  const w = 320, h = 30, x = d3.scaleLinear().domain([T0, T1]).range([4, w - 4]);
  const svg = d3.select(svgNode).attr('viewBox', `0 0 ${w} ${h}`).attr('preserveAspectRatio', 'none');
  svg.selectAll('*').remove();
  DYN.forEach((d, i) => { if (i % 2 === 0) svg.append('rect').attr('class', 'dyn-band').attr('x', x(Math.max(T0, d.a))).attr('width', x(Math.min(T1, d.b)) - x(Math.max(T0, d.a))).attr('y', 0).attr('height', h); });
  svg.append('line').attr('x1', 4).attr('x2', w - 4).attr('y1', h / 2).attr('y2', h / 2).attr('stroke', css('--rule'));
  segs.forEach(s => {
    if (s.a == null) return;
    const a = Math.max(T0, Math.min(s.a, s.b)), b = Math.min(T1, Math.max(s.a, s.b));
    svg.append('rect').attr('x', x(a)).attr('width', Math.max(3, x(b) - x(a))).attr('y', h / 2 - s.w / 2).attr('height', s.w).attr('rx', s.w / 2).attr('fill', s.c).attr('opacity', s.o ?? 1);
    if (s.dot != null) svg.append('circle').attr('cx', x(Math.max(T0, Math.min(T1, s.dot)))).attr('cy', h / 2).attr('r', 4.5).attr('fill', s.c).attr('stroke', css('--paper-2')).attr('stroke-width', 2);
  });
}
function markSpans(s) {   // '…【late words】…' -> text nodes + <mark>
  const frag = document.createDocumentFragment();
  s.split(/(【[^】]*】)/).forEach(p => { if (p.startsWith('【')) frag.appendChild(el('mark', null, p.slice(1, -1))); else if (p) frag.appendChild(document.createTextNode(p)); });
  return frag;
}
function cases() {
  const C = DATA.cases, order = ['confirmed', 'disputed', 'new', 'catalogue', 'model_explained', 'model_unexplained'];
  let f = 'all', shown = 24;
  const box = document.getElementById('cases'), fl = document.getElementById('case-filters');
  function chips() {
    fl.replaceChildren();
    ['all', ...order].forEach(k => {
      const n = k === 'all' ? C.length : C.filter(c => c.cls === k).length;
      const b = el('button', 'chip', ui().cls[k]); b.appendChild(el('span', 'n', n)); b.setAttribute('aria-pressed', f === k);
      b.addEventListener('click', () => { f = k; shown = 24; chips(); draw(); }); fl.appendChild(b);
    });
  }
  function draw() {
    box.replaceChildren();
    const L = C.filter(c => f === 'all' || c.cls === f).sort((a, b) => order.indexOf(a.cls) - order.indexOf(b.cls));
    L.slice(0, shown).forEach(c => {
      const card = el('article', 'case');
      card.appendChild(el('div', 'tag')).append(Object.assign(el('b', null, ui().cls[c.cls]), {}), ` · ${ui().side[c.side]}`);
      card.appendChild(el('h4', null, tx(c.title)));
      const cat = parseRange(c.catalogue);
      const mini = el('div', 'mini'); const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); mini.appendChild(s); card.appendChild(mini);
      miniBar(s, [cat ? { a: cat[0], b: cat[1], c: css('--neutral'), w: 10, o: .75 } : {},
                  c.q10 != null ? { a: c.q10, b: c.q90, c: css('--model'), w: 3, dot: c.model } : (c.model != null ? { a: c.model, b: c.model, c: css('--model'), w: 3, dot: c.model } : {})]);
      const ln = el('div', 'tag'); ln.append(`${ui().catalogue} ${cat ? rangeTxt(cat[0], cat[1]) : tx(c.catalogue)}  ·  `, Object.assign(el('b', null, `${ui().model} ${fy(c.model, true)}`)));
      card.appendChild(ln);
      if (c.scholarly) card.appendChild(el('div', 'note', tx(c.scholarly)));
      if (c.spans && c.spans.length) {
        const sp = el('div', 'spans'); sp.appendChild(el('div', 'tag', c.side === 'late' ? ui().spans_h : ui().spans_h_early));
        c.spans.slice(0, 3).forEach(t => { const d = el('div'); d.appendChild(markSpans(tx(t))); sp.appendChild(d); }); card.appendChild(sp);
      }
      if (c.note) card.appendChild(el('div', 'more', tx(c.note)));
      box.appendChild(card);
    });
    if (L.length > shown) { const b = el('button', 'btn ghost', `+ ${L.length - shown}`); b.addEventListener('click', () => { shown += 24; draw(); }); box.appendChild(b); }
  }
  renderers.push(() => { chips(); draw(); }); chips(); draw();
}

/* ================= IMITATION ================= */
function imitation() {
  const I = DATA.imit, box = document.getElementById('pull-chart');
  function draw() {
    const rows = I.human.map(d => ({ name: `${ui().human} · ${d.bin} ${ui().chars}`, kind: 'h', ...d })).concat(I.ai.map(d => ({ name: d.model, kind: 'a', ...d })));
    const w = W(box), rh = 30, m = { l: isZh() ? 150 : 170, r: 20, t: 10, b: 34 }, h = m.t + m.b + rows.length * rh;
    const svg = svgIn(box, w, h), x = d3.scaleLinear().domain([0, 0.2]).range([m.l, w - m.r]);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(4, '.0%').tickSize(-(h - m.t - m.b)))
      .call(g => g.selectAll('.tick line').attr('stroke', css('--rule')).attr('stroke-dasharray', '2 3'));
    svg.append('text').attr('class', 'lbl').attr('x', (m.l + w - m.r) / 2).attr('y', h - 4).attr('text-anchor', 'middle').text(ui().pull);
    const c = (d) => d.kind === 'h' ? css('--scholar') : css('--third');
    rows.forEach((d, i) => {
      const yy = m.t + i * rh + rh / 2;
      if (i === I.human.length) svg.append('line').attr('x1', 0).attr('x2', w - m.r).attr('y1', yy - rh / 2).attr('y2', yy - rh / 2).attr('stroke', css('--rule'));
      svg.append('text').attr('class', 'lbl').attr('x', m.l - 10).attr('y', yy + 4).attr('text-anchor', 'end').style('fill', css('--ink-2')).style('font-size', '13px').text(d.name);
      if (d.lo != null) svg.append('line').attr('x1', x(d.lo)).attr('x2', x(d.hi)).attr('y1', yy).attr('y2', yy).attr('stroke', c(d)).attr('stroke-width', 2).attr('stroke-linecap', 'round');
      svg.append('circle').attr('cx', x(d.pull)).attr('cy', yy).attr('r', 5.5).attr('fill', c(d)).attr('stroke', css('--paper')).attr('stroke-width', 2);
      svg.append('rect').attr('x', 0).attr('y', yy - rh / 2).attr('width', w).attr('height', rh).attr('fill', 'transparent')
        .on('pointermove', (ev) => showTip(ev, [{ v: d3.format('.1%')(d.pull), l: d.name }].concat(d.lo != null ? [{ l: `95%: ${d3.format('.1%')(d.lo)}–${d3.format('.1%')(d.hi)}` }] : [])))
        .on('pointerleave', hideTip);
    });
  }
  const gal = document.getElementById('gallery');
  function drawGallery() {
    gal.replaceChildren();
    I.gallery.forEach(p => {
      const card = el('figure', 'poem'); card.style.margin = 0;
      card.appendChild(el('div', 'ptext', tx(p.text.replace(/／/g, ''))));
      const meta = el('figcaption', 'pmeta');
      meta.append(`${p.model} · ${ui().asked(isZh() ? tx(p.target_author) : ({ '李白': 'Li Bai', '杜甫': 'Du Fu', '王維': 'Wang Wei' }[p.target_author] || p.target_author))}`, el('br'), `${ui().reads} `, el('b', null, fy(p.apparent, true)));
      const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      card.append(meta, s);
      miniBar(s, [{ a: p.target - 30, b: p.target + 30, c: css('--scholar'), w: 8, o: .8 }, { a: p.apparent, b: p.apparent, c: css('--third'), w: 3, dot: p.apparent }]);
      gal.appendChild(card);
    });
  }
  renderers.push(() => { draw(); drawGallery(); }); draw(); drawGallery();
}

/* ================= FREEZE ================= */
function freeze() {
  const F = DATA.freeze.ratio, box = document.getElementById('freeze-chart');
  function draw() {
    const w = W(box), h = 300, m = { l: 46, r: 16, t: 30, b: 30 };
    const svg = svgIn(box, w, h), x = timeScale(w, m), y = d3.scaleLinear().domain([0.7, 1.7]).range([h - m.b, m.t]);
    dynBands(svg, x, m.t, h - m.b, true, m.t - 10);
    timeAxis(svg, x, h - m.b);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(w - m.l - m.r)))
      .call(g => g.selectAll('.tick line').attr('stroke', css('--rule')).attr('stroke-dasharray', '2 3')).call(g => g.select('.domain').remove());
    svg.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(1)).attr('y2', y(1)).attr('stroke', css('--ink-3'));
    svg.append('text').attr('class', 'lbl').attr('x', w - m.r - 4).attr('y', y(1) + 14).attr('text-anchor', 'end').text(isZh() ? '語法與詞彙一樣可斷代' : 'grammar as datable as vocabulary');
    svg.append('path').attr('d', d3.line().x(d => x(d.mid)).y(d => y(d.r)).curve(d3.curveMonotoneX)(F)).attr('fill', 'none').attr('stroke', css('--model')).attr('stroke-width', 2.2);
    svg.selectAll('circle').data(F).join('circle').attr('cx', d => x(d.mid)).attr('cy', d => y(d.r)).attr('r', 5).attr('fill', css('--model')).attr('stroke', css('--paper')).attr('stroke-width', 2);
    const pk = F.reduce((a, b) => a.r > b.r ? a : b);
    svg.append('text').attr('class', 'lbl').attr('x', x(pk.mid) + 10).attr('y', y(pk.r) - 8).style('fill', css('--ink')).style('font-size', '13px')
      .text(isZh() ? '古文運動' : 'Ancient-Prose movement');
    svg.selectAll('rect.hit').data(F).join('rect').attr('x', d => x(d.mid) - 16).attr('width', 32).attr('y', m.t).attr('height', h - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', (ev, d) => showTip(ev, [{ v: d.r.toFixed(2), l: isZh() ? tx(d.period) : d.en }, { l: ui().ratio }])).on('pointerleave', hideTip);
  }
  renderers.push(draw); draw();
}

/* ================= FORGERY ================= */
function blind() {
  const B = DATA.blind; let f = 'all';
  const box = document.getElementById('blind-chart'), fl = document.getElementById('blind-filters');
  function chips() { fl.replaceChildren(); ['all', 'A', 'B', 'D'].forEach(k => { const b = el('button', 'chip', ui().blind_cat[k]); b.setAttribute('aria-pressed', f === k); b.addEventListener('click', () => { f = k; chips(); draw(); }); fl.appendChild(b); }); }
  function draw() {
    const rows = B.works.filter(d => f === 'all' || d.cat === f).sort((a, b) => a.cat.localeCompare(b.cat) || (a.cons[0] - b.cons[0]));
    const w = W(box), rh = 24, m = { l: 112, r: 14, t: 26, b: 28 }, h = m.t + m.b + rows.length * rh;
    const svg = svgIn(box, w, h), x = d3.scaleLinear().domain([-1100, T1]).range([m.l, w - m.r]);
    DYN.concat([{ a: -1100, b: -801, zh: '商周', en: 'Shang–Zhou' }]).forEach((d, i) => { const a = Math.max(-1100, d.a), c = Math.min(T1, d.b);
      if (i % 2 === 0) svg.append('rect').attr('class', 'dyn-band').attr('x', x(a)).attr('width', Math.max(0, x(c) - x(a))).attr('y', m.t).attr('height', h - m.b - m.t);
      if (fits(d, x(c) - x(a))) svg.append('text').attr('class', 'dyn-lbl').attr('x', (x(a) + x(c)) / 2).attr('y', m.t - 10).attr('text-anchor', 'middle').text(dynName(d)); });
    timeAxis(svg, x, h - m.b, [-1000, -500, 0, 500, 1000, 1500]);
    let last = null;
    rows.forEach((d, i) => {
      const yy = m.t + i * rh + rh / 2;
      if (last && last !== d.cat) svg.append('line').attr('x1', 0).attr('x2', w - m.r).attr('y1', yy - rh / 2).attr('y2', yy - rh / 2).attr('stroke', css('--ink-3')).attr('stroke-dasharray', '1 3');
      last = d.cat;
      svg.append('text').attr('x', m.l - 10).attr('y', yy + 5).attr('text-anchor', 'end').style('font-family', 'var(--serif-zh)').style('font-size', '15px').style('fill', css('--ink')).text(tx(d.work));
      const bar = (a, b, c, th, o) => svg.append('rect').attr('x', x(Math.max(-1100, a))).attr('width', Math.max(3, x(Math.min(T1, b)) - x(Math.max(-1100, a)))).attr('y', yy - th / 2).attr('height', th).attr('rx', th / 2).attr('fill', c).attr('opacity', o);
      bar(d.attr[0], d.attr[1], css('--neutral'), 9, .6);
      bar(d.cons[0], d.cons[1], css('--scholar'), 9, .75);
      svg.append('line').attr('x1', x(d.q10)).attr('x2', x(d.q90)).attr('y1', yy).attr('y2', yy).attr('stroke', css('--model')).attr('stroke-width', 2);
      svg.append('circle').attr('cx', x(d.median)).attr('cy', yy).attr('r', 5).attr('fill', css('--model')).attr('stroke', css('--paper')).attr('stroke-width', 2);
      svg.append('rect').attr('x', 0).attr('y', yy - rh / 2).attr('width', w).attr('height', rh).attr('fill', 'transparent')
        .on('pointermove', (ev) => showTip(ev, [{ v: tx(d.work), l: ui().blind_cat[d.cat] }, { c: css('--neutral'), v: rangeTxt(d.attr[0], d.attr[1]), l: ui().attributed },
          { c: css('--scholar'), v: rangeTxt(d.cons[0], d.cons[1]), l: ui().consensus }, { c: css('--model'), v: fy(d.median, true), l: `${ui().model} (${fy(d.q10)}–${fy(d.q90)})` }]))
        .on('pointerleave', hideTip);
    });
  }
  function strip(node, data, groups, colorOf, label) {
    const w = W(node), rh = 44, m = { l: isZh() ? 84 : 140, r: 14, t: 8, b: 28 }, h = m.t + m.b + groups.length * rh;
    const svg = svgIn(node, w, h), x = d3.scaleLinear().domain([-800, T1]).range([m.l, w - m.r]);
    dynBands(svg, x, m.t, h - m.b, false);
    timeAxis(svg, x, h - m.b, [-500, 0, 500, 1000, 1500]);
    groups.forEach((g, gi) => {
      const yy = m.t + gi * rh + rh / 2;
      svg.append('text').attr('class', 'lbl').attr('x', m.l - 10).attr('y', yy + 4).attr('text-anchor', 'end').style('fill', css('--ink-2')).style('font-size', '13px').text(ui()[g]);
      const D = data.filter(d => d.g === g);
      const med = d3.median(D, d => d.median);
      svg.append('line').attr('x1', x(med)).attr('x2', x(med)).attr('y1', yy - 16).attr('y2', yy + 16).attr('stroke', colorOf(g)).attr('stroke-width', 2.5);
      D.forEach((d, i) => {
        const jy = yy + ((i * 7919) % 25 - 12) * .9;
        svg.append('circle').attr('cx', x(Math.max(-800, Math.min(T1, d.median)))).attr('cy', jy).attr('r', 4).attr('fill', colorOf(g)).attr('fill-opacity', .7).attr('stroke', css('--paper')).attr('stroke-width', 1)
          .on('pointermove', (ev) => showTip(ev, [{ v: fy(d.median, true), l: label(d) }])).on('pointerleave', hideTip);
      });
    });
  }
  function drawStrips() {
    const col = { guwen: css('--model'), jinwen: css('--scholar'), zhougao: css('--third'), unique: css('--model'), parallel: css('--scholar'), mixed: css('--third') };
    strip(document.getElementById('shang-chart'), B.shangshu.map(d => ({ ...d, g: d.cls })), ['jinwen', 'zhougao', 'guwen'], g => col[g], d => tx(d.title));
    strip(document.getElementById('liezi-chart'), B.liezi.map(d => ({ ...d, g: d.kind })), ['parallel', 'mixed', 'unique'], g => col[g], d => tx(d.text) + '…');
  }
  renderers.push(() => { chips(); draw(); drawStrips(); }); chips(); draw(); drawStrips();
}

/* ================= UNDATED ================= */
function undated() {
  const N = DATA.newdates; let tier = 'A', q = '';
  const list = document.getElementById('ud-list'), tiers = document.getElementById('ud-tiers'), input = document.getElementById('ud-search');
  const ax = document.getElementById('ud-axis');
  function chips() { tiers.replaceChildren(); ['A', 'B', 'C', 'all'].forEach(k => { const n = k === 'all' ? N.length : N.filter(d => d.tier === k).length;
    const b = el('button', 'chip', ui().tiers[k]); b.appendChild(el('span', 'n', n)); b.setAttribute('aria-pressed', tier === k); b.addEventListener('click', () => { tier = k; chips(); draw(); }); tiers.appendChild(b); }); }
  function axis() {
    const w = W(list), sample = list.querySelector('.ud-row svg');
    const svg = svgIn(ax, w, 34);
    // align with the row's bar column when rows exist
    let l = 0, r = w;
    if (sample) { const a = sample.getBoundingClientRect(), b = ax.getBoundingClientRect(); l = a.left - b.left; r = a.right - b.left; }
    const x = d3.scaleLinear().domain([T0, T1]).range([l + 4, r - 4]);
    dynBands(svg, x, 0, 34, true, 14);
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(0,20)').call(d3.axisBottom(x).tickValues([-500, 0, 500, 1000, 1500]).tickFormat(d => fy(d)).tickSize(3));
  }
  function draw() {
    const qt = q ? [...q].map(c => (window.S2T && S2T[c]) || c).join('') : '';
    const L = N.filter(d => (tier === 'all' || d.tier === tier) && (!q || d.title.includes(q) || d.title.includes(qt) || t2s(d.title).includes(q) || (d.src || '').includes(q) || d.id.includes(q)))
      .sort((a, b) => a.median - b.median);
    list.replaceChildren();
    if (!L.length) list.appendChild(el('div', 'ud-row', ui().none));
    L.slice(0, 250).forEach(d => {
      const r = el('div', 'ud-row'); r.setAttribute('role', 'listitem');
      const t = el('div', 't', tx(d.title)); t.title = tx(d.title);
      d.flags.forEach(fk => t.appendChild(el('span', 'flag', ui().flags[fk])));
      t.appendChild(el('span', 's', `${d.src || d.id} · ${d.n.toLocaleString()} ${ui().chars}`));
      const s = document.createElementNS('http://www.w3.org/2000/svg', 's' + 'vg');
      const y = el('div', 'y', fy(d.median, true)); y.appendChild(el('small', null, `${fy(d.q10)}–${fy(d.q90)}`));
      r.append(t, s, y); list.appendChild(r);
      miniBar(s, [{ a: d.q10, b: d.q90, c: css('--model'), w: 2, o: .7 }, { a: d.q25, b: d.q75, c: css('--model'), w: 7, o: .55, dot: d.median }]);
    });
    if (L.length > 250) list.appendChild(el('div', 'ud-row muted', ui().more(L.length - 250)));
    axis();
  }
  input.addEventListener('input', () => { q = input.value.trim(); draw(); });
  renderers.push(() => { chips(); draw(); }); chips(); draw();
}

/* ================= side rail ================= */
function rail() {
  const links = [...document.querySelectorAll('.rail-nav a')], dot = document.getElementById('rail-dot');
  const secs = links.map(a => document.querySelector(a.getAttribute('href')));
  function update() {
    const y = scrollY + innerHeight * .35;
    let k = 0; secs.forEach((s, i) => { if (s && s.offsetTop <= y) k = i; });
    links.forEach((a, i) => a.classList.toggle('on', i === k));
    const a = links[k]; if (a && dot) dot.style.top = (a.offsetTop + a.offsetHeight / 2 - 5) + 'px';
  }
  addEventListener('scroll', update, { passive: true }); update(); renderers.push(update);
}

/* ================= boot ================= */
async function boot() {
  applyLang();
  const B = window.RUOQU_DATA || {};
  const get = (f) => B[f] ? Promise.resolve(B[f]) : fetch(`data/${f}.json`).then(r => r.json());
  [DATA.quiz, DATA.life, DATA.eras, DATA.scatter, DATA.cases, DATA.imit, DATA.freeze, DATA.blind, DATA.newdates] =
    await Promise.all(['quiz', 'lifelines', 'eras', 'scatter', 'cases', 'imitation', 'freeze', 'blind', 'newdates'].map(get));
  for (const f of [hero, quiz, words, scatter, cases, imitation, freeze, blind, undated, rail]) { try { f(); } catch (e) { console.error(f.name, e); } }
}
boot();
})();
