import { get } from './api.js';
import { $, h, clear, pct, num, notice } from './ui.js';
import { hbars, confusionMatrix } from './charts.js';

const METRICS = [
  ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1-score'], ['error_rate', 'Error rate'],
];
let loaded = false, data = null;
const ui = { metric: 'accuracy', sortBy: 'accuracy', sortDir: -1 };

export async function onShow() {
  if (loaded) return;
  const out = $('#perfOut');
  try {
    data = await get('/api/results');
    loaded = true;
    render(out);
  } catch (err) {
    clear(out).append(notice(err.message));
  }
}

const fmt = (v) => `${Number(v).toFixed(2)}%`;

function render(out) {
  clear(out);
  const chartBox = h('div', {});
  const noteBox = h('p', { class: 'axis-note' });
  const metricSelect = h('select', { 'aria-label': 'Metric', onchange: (e) => { ui.metric = e.target.value; drawChart(); } },
    METRICS.map(([key, label]) => h('option', { value: key }, label)));

  function drawChart() {
    const lowerIsBetter = ui.metric === 'error_rate';
    const rows = [...data.rows].sort((a, b) => (lowerIsBetter ? a[ui.metric] - b[ui.metric] : b[ui.metric] - a[ui.metric]));
    const values = rows.map((r) => r[ui.metric]);
    const lo = lowerIsBetter ? 0 : Math.max(0, Math.floor(Math.min(...values) - 2));
    const hi = lowerIsBetter ? Math.ceil(Math.max(...values)) + 1 : 100;
    hbars(chartBox, rows.map((r, i) => ({
      label: r.algorithm, frac: (r[ui.metric] - lo) / (hi - lo), text: fmt(r[ui.metric]),
      model: r.live ? r.key : 'other', top: i === 0, swatch: true,
    })), { wide: true });
    noteBox.textContent = lowerIsBetter ? 'Shorter bars are better.' : `Bars start at ${lo}%, not zero, so small differences are visible.`;
  }

  out.append(h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'Algorithm comparison'), metricSelect),
    h('p', { class: 'panel-note' }, 'Results recorded in notebook 08 on the 8,400 held-out images. Coloured bars are models this app runs. Grey bars have no loaded model here: Random Forest and Logistic Regression only exist as results in your comparison notebook.'),
    chartBox, noteBox));
  drawChart();

  out.append(tablePanel());
  out.append(detailPanel());
}

function tablePanel() {
  const tbody = h('tbody', {});
  const cols = [['algorithm', 'Algorithm'], ...METRICS];
  const head = h('tr', {});

  function draw() {
    const rows = [...data.rows].sort((a, b) => {
      const x = a[ui.sortBy], y = b[ui.sortBy];
      return (typeof x === 'string' ? x.localeCompare(y) : x - y) * ui.sortDir;
    });
    const best = Math.max(...data.rows.map((r) => r.accuracy));
    clear(head).append(...cols.map(([key, label]) => h('th', {
      class: 'sortable', 'aria-sort': ui.sortBy === key ? (ui.sortDir > 0 ? 'ascending' : 'descending') : 'none',
      onclick: () => { ui.sortDir = ui.sortBy === key ? -ui.sortDir : (key === 'algorithm' ? 1 : -1); ui.sortBy = key; draw(); },
    }, label + (ui.sortBy === key ? (ui.sortDir > 0 ? ' (asc)' : ' (desc)') : ''))));
    clear(tbody);
    for (const r of rows) {
      tbody.append(h('tr', { class: r.accuracy === best ? 'best' : '' },
        h('td', {}, r.algorithm, h('span', { class: `tag ${r.live ? 'live' : r.key ? 'off' : ''}` }, r.live ? 'Runs in this app' : r.key ? 'Model not loaded' : 'Results only')),
        ...METRICS.map(([key]) => h('td', {}, fmt(r[key])))));
    }
  }
  draw();
  return h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'All metrics')),
    h('p', { class: 'panel-note' }, 'Precision, recall and F1 are weighted averages across the ten digits, as in your notebooks. Click a column heading to sort.'),
    h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, head), tbody)));
}

function detailPanel() {
  const panel = h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Per-digit results')));
  const detail = data.detail;
  if (!detail || !detail.models || !Object.keys(detail.models).length) {
    panel.append(notice('Per-digit reports and confusion matrices are not available yet. Put train.csv in the data folder, then run python -m ml.evaluate. It re-creates your 80/20 split (random_state=42), scores the saved models and writes results/metrics.json. Reload this page afterwards.', 'info'));
    return panel;
  }

  const labels = Object.fromEntries(data.rows.filter((r) => r.key).map((r) => [r.key, r.algorithm]));
  const select = h('select', { 'aria-label': 'Model', onchange: () => draw() },
    Object.keys(detail.models).map((k) => h('option', { value: k }, labels[k] || k)));
  panel.querySelector('.panel-head').append(select);
  const body = h('div', {});
  panel.append(h('p', { class: 'panel-note' }, `Computed on ${num(detail.split.test)} held-out images (20% of train.csv, random_state=42) on ${new Date(detail.generated).toLocaleString()}.`), body);

  function draw() {
    const m = detail.models[select.value];
    const cm = h('div', { class: 'cm-wrap' });
    const tbody = h('tbody', {});
    for (let d = 0; d < 10; d++) {
      const r = m.report[String(d)];
      tbody.append(h('tr', {}, h('td', {}, `Digit ${d}`), h('td', {}, pct(r.precision)), h('td', {}, pct(r.recall)), h('td', {}, pct(r['f1-score'])), h('td', {}, num(r.support))));
    }
    const w = m.report['weighted avg'];
    tbody.append(h('tr', { class: 'best' }, h('td', {}, 'Weighted average'), h('td', {}, pct(w.precision)), h('td', {}, pct(w.recall)), h('td', {}, pct(w['f1-score'])), h('td', {}, num(w.support))));

    clear(body).append(
      h('dl', { class: 'facts', style: 'margin-bottom:20px' },
        h('div', {}, h('dt', {}, 'Accuracy'), h('dd', {}, pct(m.accuracy))),
        h('div', {}, h('dt', {}, 'Correct'), h('dd', {}, num(m.correct))),
        h('div', {}, h('dt', {}, 'Mistakes'), h('dd', {}, num(m.incorrect))),
        h('div', {}, h('dt', {}, 'Images scored'), h('dd', {}, num(m.total)))),
      h('div', { class: 'grid even' },
        h('div', {}, h('h3', { style: 'margin-bottom:10px' }, 'Precision, recall and F1 by digit'),
          h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Digit', 'Precision', 'Recall', 'F1-score', 'Images'].map((t) => h('th', {}, t)))), tbody))),
        h('div', {}, h('h3', { style: 'margin-bottom:10px' }, 'Confusion matrix'), cm,
          h('p', { class: 'axis-note' }, 'Rows are the actual digit and columns the predicted digit. Blue cells are correct, red cells are mistakes.'))));
    confusionMatrix(cm, m.confusion);
  }
  draw();
  return panel;
}
