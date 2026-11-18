import { get } from './api.js';
import { $, h, clear, notice } from './ui.js';

let loaded = false;

export async function onShow() {
  if (loaded) return;
  const out = $('#modelCards');
  try {
    const { models } = await get('/api/models');
    loaded = true;
    clear(out).append(...models.map(card));
  } catch (err) {
    clear(out).append(notice(err.message));
  }
}

function card(m) {
  const el = h('div', { class: `panel model-card m-${m.key}` },
    h('h2', {}, m.label, h('span', { class: `tag ${m.available ? 'live' : 'off'}` }, m.available ? 'Loaded' : 'Not loaded')),
    h('p', { class: 'sum' }, m.summary));

  if (!m.available) {
    el.append(notice(`${m.error || 'Model file not found.'} Run notebook ${m.notebook} to create ${m.file}, copy it into the models folder, then restart the server. Or train it with python -m ml.train --models ${m.key}.`, 'warn'));
    return el;
  }

  const kv = h('dl', { class: 'kv' });
  const add = (k, v) => kv.append(h('dt', {}, k), h('dd', {}, String(v)));
  add('Model file', m.file);
  add('File size', `${m.size_mb} MB`);
  add('Notebook', m.notebook);
  Object.entries(m.params).forEach(([k, v]) => add(k, v));

  if (m.key === 'mlp' && m.params['Layer sizes']) {
    const sizes = m.params['Layer sizes'].split(' to ');
    const chain = h('div', { class: 'chain', 'aria-label': 'Layer sizes' });
    sizes.forEach((s, i) => {
      if (i) chain.append(h('span', { class: 'link' }));
      chain.append(h('span', { class: 'node' }, s));
    });
    el.append(chain, h('p', { class: 'hint', style: 'margin:0 0 14px' }, 'Neurons per layer, from input to output.'));
  }
  el.append(kv);

  if (m.layers) {
    el.append(h('div', { class: 'table-wrap layer-table' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Layer', 'Output shape', 'Parameters'].map((t) => h('th', {}, t)))),
      h('tbody', {}, m.layers.map((l) => h('tr', {}, h('td', {}, l.type), h('td', {}, l.output), h('td', {}, l.params.toLocaleString())))))));
  }
  return el;
}
