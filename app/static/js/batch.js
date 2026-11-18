import { api } from './api.js';
import { $, $$, h, clear, drawDigit, pct, num, notice, toast } from './ui.js';
import { groupedBars } from './charts.js';

const PAGE_SIZE = 25;
let statusInfo = null;

export function init(status) {
  statusInfo = status;
  const models = $('#batchModels');
  for (const m of status.models) {
    models.append(h('label', { class: `check m-${m.key} ${m.available ? '' : 'off'}`, title: m.available ? '' : m.error },
      h('input', { type: 'checkbox', value: m.key, checked: m.available, disabled: !m.available }), m.label));
  }
  $('#batchCount').max = status.max_batch_rows;
  $('#batchHint').textContent = `Up to ${num(status.max_batch_rows)} rows per run. SVM and kNN take longer than MLP and CNN.`;

  if (!status.test_csv) {
    $('#batchSource option[value="bundled"]').disabled = true;
    $('#batchSource').value = 'upload';
  }
  $('#batchSource').addEventListener('change', syncSource);
  $('#btnBatch').addEventListener('click', run);
  syncSource();
}

export function onShow() {}

function syncSource() {
  const source = $('#batchSource').value;
  $$('[data-src]').forEach((el) => el.classList.toggle('hidden', el.dataset.src !== source));
}

async function run() {
  const out = clear($('#batchOut'));
  const keys = $$('#batchModels input:checked').map((i) => i.value);
  if (!keys.length) return out.append(notice('Select at least one model.'));

  const form = new FormData();
  const source = $('#batchSource').value;
  form.append('source', source);
  form.append('models', keys.join(','));
  if (source === 'bundled') {
    const start = Number($('#batchStart').value), count = Number($('#batchCount').value);
    if (!Number.isInteger(start) || start < 0) return out.append(notice('Start row must be a whole number, 0 or higher.'));
    if (!Number.isInteger(count) || count < 1 || count > statusInfo.max_batch_rows) {
      return out.append(notice(`Number of rows must be between 1 and ${num(statusInfo.max_batch_rows)}.`));
    }
    form.append('start', start);
    form.append('count', count);
  } else {
    const file = $('#batchFile').files[0];
    if (!file) return out.append(notice('Choose a CSV file first.'));
    form.append('file', file);
  }

  const button = $('#btnBatch');
  button.disabled = true;
  out.append(h('div', { class: 'panel loading', style: 'min-height:120px' }, h('p', { class: 'kicker' }, 'Running the models. Large batches can take a minute.')));
  try {
    render(out, await api('/api/batch', { method: 'POST', body: form }));
  } catch (err) {
    clear(out).append(notice(err.message));
  } finally {
    button.disabled = false;
  }
}

function render(out, res) {
  clear(out);
  const labelOf = Object.fromEntries(statusInfo.models.map((m) => [m.key, m.label]));
  const stack = h('div', { class: 'stack' });
  out.append(stack);

  if (res.skipped) stack.append(notice(`${num(res.skipped)} row(s) with missing, non-numeric or invalid label values were skipped.`, 'warn'));
  if (res.truncated) stack.append(notice(`Only the first ${num(res.cap)} rows of the file were used.`, 'warn'));

  // Summary
  const facts = h('dl', { class: 'facts' }, fact('Rows processed', num(res.n)));
  if (res.agreement !== null) facts.append(fact('All models agree on', pct(res.agreement, 1)));
  for (const key of res.models) {
    if (res.metrics) {
      const m = res.metrics[key];
      facts.append(fact(`${labelOf[key]} accuracy`, pct(m.accuracy, 2), `${num(m.correct)} of ${num(m.total)}`));
    }
    facts.append(fact(`${labelOf[key]} time`, `${res.timing[key]} s`));
  }
  stack.append(h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Summary')), facts));

  // Distribution
  const chart = h('div', {});
  stack.append(h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'How many images each model assigned to each digit')), chart));
  groupedBars(chart, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    res.models.map((key) => ({ key, label: labelOf[key], values: res.distribution[key] })));

  // Gallery
  const gallery = h('div', { class: 'gallery' });
  res.gallery.forEach((pixels, i) => {
    const preds = res.models.map((k) => res.predictions[k][i]);
    const disagree = new Set(preds).size > 1;
    const canvas = h('canvas', {});
    drawDigit(canvas, pixels);
    gallery.append(h('figure', { class: disagree ? 'diff' : '', title: `Row ${res.ids[i]}` },
      h('div', { class: 'pic' }, canvas),
      h('figcaption', {}, res.models.map((k, j) => h('span', { class: `m-${k}`, title: labelOf[k] }, preds[j])))));
  });
  stack.append(h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, `First ${res.gallery.length} images`)),
    h('p', { class: 'panel-note' }, 'Numbers under each image are the predictions, coloured by model. A red outline means the models disagree.'),
    gallery,
    h('div', { class: 'legend' }, res.models.map((k) => h('span', { class: `m-${k}` }, h('i', { class: 'swatch' }), labelOf[k])))));

  stack.append(tablePanel(res, labelOf));
}

function fact(label, value, small) {
  return h('div', {}, h('dt', {}, label), h('dd', {}, value, small ? h('small', {}, ` ${small}`) : null));
}

// Predictions table with filter, pagination and CSV export
function tablePanel(res, labelOf) {
  const hasLabels = Array.isArray(res.labels);
  const multi = res.models.length > 1;
  const state = { page: 0, filter: 'all' };

  const isDisagree = (i) => multi && new Set(res.models.map((k) => res.predictions[k][i])).size > 1;
  const isWrong = (i) => hasLabels && res.models.some((k) => res.predictions[k][i] !== res.labels[i]);
  const matching = () => res.ids.map((_, i) => i).filter((i) =>
    state.filter === 'all' || (state.filter === 'disagree' && isDisagree(i)) || (state.filter === 'wrong' && isWrong(i)));

  const body = h('tbody', {}), pager = h('div', { class: 'pagination' });
  const head = h('tr', {}, h('th', {}, 'Row'), hasLabels ? h('th', {}, 'Actual') : null, res.models.map((k) => h('th', {}, labelOf[k])));

  const filter = h('select', { 'aria-label': 'Filter rows', onchange: (e) => { state.filter = e.target.value; state.page = 0; draw(); } },
    h('option', { value: 'all' }, 'All rows'),
    multi ? h('option', { value: 'disagree' }, 'Models disagree') : null,
    hasLabels ? h('option', { value: 'wrong' }, 'At least one wrong') : null);

  function draw() {
    const rows = matching(), pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    clear(body);
    for (const i of rows.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE)) {
      body.append(h('tr', { class: isDisagree(i) ? 'diff' : '' },
        h('td', {}, res.ids[i]),
        hasLabels ? h('td', {}, res.labels[i]) : null,
        res.models.map((k) => {
          const wrong = hasLabels && res.predictions[k][i] !== res.labels[i];
          return h('td', { style: wrong ? 'color:var(--danger);font-weight:700' : '' }, res.predictions[k][i]);
        })));
    }
    if (!rows.length) body.append(h('tr', {}, h('td', { colspan: head.children.length, style: 'text-align:center;color:var(--muted)' }, 'No rows match this filter.')));
    clear(pager).append(
      h('button', { class: 'btn', disabled: state.page === 0, onclick: () => { state.page--; draw(); } }, 'Previous'),
      h('span', { class: 'kicker' }, `${num(rows.length)} rows, page ${state.page + 1} of ${pages}`),
      h('button', { class: 'btn', disabled: state.page >= pages - 1, onclick: () => { state.page++; draw(); } }, 'Next'));
  }

  const panel = h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'Predictions'),
      h('div', { class: 'row' }, filter, h('button', { class: 'btn', onclick: () => downloadCsv(res, labelOf) }, 'Download CSV'))),
    h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, head), body)), pager);
  draw();
  return panel;
}

function downloadCsv(res, labelOf) {
  const hasLabels = Array.isArray(res.labels);
  const lines = [['ImageId', ...(hasLabels ? ['Actual'] : []), ...res.models.map((k) => labelOf[k])].join(',')];
  res.ids.forEach((id, i) => {
    lines.push([id, ...(hasLabels ? [res.labels[i]] : []), ...res.models.map((k) => res.predictions[k][i])].join(','));
  });
  const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
  const link = h('a', { href: url, download: 'predictions.csv' });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  toast('Predictions downloaded');
}
