import { h, svg, clear, num } from './ui.js';

// Horizontal bars. rows: [{label, frac (0-1), text, model?, top?, dim?}]
export function hbars(container, rows, { wide = false } = {}) {
  clear(container);
  container.classList.add('hbars');
  container.classList.toggle('wide', wide);
  const fills = [];
  for (const r of rows) {
    const fill = h('span', { style: 'width:0%' });
    fills.push([fill, Math.max(0, Math.min(1, r.frac)) * 100]);
    container.append(h('div', { class: `hb ${r.top ? 'top' : ''} ${r.dim ? 'dim' : ''} m-${r.model || 'other'}` },
      h('div', { class: 'lab' }, r.swatch ? h('i', { class: 'swatch' }) : null, r.label),
      h('div', { class: 'bar' }, fill),
      h('div', { class: 'val' }, r.text)));
  }
  requestAnimationFrame(() => fills.forEach(([el, w]) => { el.style.width = `${w}%`; }));
}

// Grouped vertical bars: categories on x, one bar per series. series: [{key, label, values}]
export function groupedBars(container, categories, series) {
  clear(container);
  const W = 640, H = 250, L = 46, R = 8, T = 10, B = 28;
  const plotW = W - L - R, plotH = H - T - B;
  const rawMax = Math.max(1, ...series.flatMap((s) => s.values));
  const step = niceStep(rawMax / 4);
  const max = Math.ceil(rawMax / step) * step;
  const root = svg('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img' });

  for (let v = 0; v <= max + 1e-9; v += step) {
    const y = T + plotH - (v / max) * plotH;
    root.append(svg('line', { class: 'grid-line', x1: L, x2: W - R, y1: y, y2: y }),
      svg('text', { x: L - 8, y: y + 4, 'text-anchor': 'end' }, num(v)));
  }
  const group = plotW / categories.length;
  const barW = Math.min(22, (group * 0.78) / series.length);
  categories.forEach((cat, i) => {
    const x0 = L + i * group + (group - barW * series.length) / 2;
    series.forEach((s, j) => {
      const bh = (s.values[i] / max) * plotH;
      const bar = svg('g', { class: `m-${s.key}` },
        svg('rect', { x: x0 + j * barW, y: T + plotH - bh, width: Math.max(1, barW - 2), height: bh, rx: 2, style: 'fill:var(--c)' }));
      bar.append(svg('title', {}, `${s.label}: ${num(s.values[i])} predicted as ${cat}`));
      root.append(bar);
    });
    root.append(svg('text', { x: L + i * group + group / 2, y: H - 8, 'text-anchor': 'middle' }, cat));
  });
  container.append(root, h('div', { class: 'legend' },
    series.map((s) => h('span', { class: `m-${s.key}` }, h('i', { class: 'swatch' }), s.label))));
}

function niceStep(raw) {
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pow;
}

// Confusion matrix: rows = actual digit, columns = predicted digit.
export function confusionMatrix(container, matrix) {
  clear(container);
  const cell = 44, L = 44, T = 26, n = matrix.length;
  const W = L + n * cell + 4, H = T + n * cell + 44;
  const root = svg('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img' });
  matrix.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    root.append(svg('text', { x: L - 10, y: T + i * cell + cell / 2 + 4, 'text-anchor': 'end' }, i),
      svg('text', { x: L + i * cell + cell / 2, y: T - 8, 'text-anchor': 'middle' }, i));
    row.forEach((value, j) => {
      const frac = value / total;
      const alpha = value === 0 ? 0 : 0.1 + 0.9 * Math.sqrt(frac);
      const colour = i === j ? '51,80,224' : '200,64,47';
      const rect = svg('rect', { x: L + j * cell + 1, y: T + i * cell + 1, width: cell - 2, height: cell - 2, rx: 4,
        fill: value === 0 ? '#f3f5fb' : `rgba(${colour},${alpha.toFixed(3)})` });
      rect.append(svg('title', {}, `Actual ${i}, predicted ${j}: ${num(value)}`));
      root.append(rect, svg('text', {
        class: 'cell-text', x: L + j * cell + cell / 2, y: T + i * cell + cell / 2 + 4, 'text-anchor': 'middle',
        style: `fill:${alpha > 0.5 ? '#fff' : value === 0 ? '#b3bbd3' : '#111633'}`,
      }, value));
    });
  });
  root.append(svg('text', { x: L + (n * cell) / 2, y: H - 10, 'text-anchor': 'middle' }, 'Predicted digit'),
    svg('text', { x: 0, y: 0, 'text-anchor': 'middle', transform: `translate(10 ${T + (n * cell) / 2}) rotate(-90)` }, 'Actual digit'));
  container.append(root);
}
