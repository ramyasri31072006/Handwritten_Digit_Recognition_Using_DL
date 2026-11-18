import { get } from './api.js';
import { $, $$ } from './ui.js';
import * as predict from './predict.js';
import * as batch from './batch.js';
import * as performance from './performance.js';
import * as models from './models.js';
import * as dataset from './dataset.js';

const views = { predict, batch, performance, models, dataset };

function show(name) {
  if (!views[name]) name = 'predict';
  $$('.tab').forEach((t) => t.classList.toggle('active', t.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  views[name].onShow?.();
}

function setStatus(kind, text) {
  const el = $('#status');
  el.className = `status ${kind}`;
  el.querySelector('span').textContent = text;
}

async function boot() {
  $('#tabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.tab');
    if (tab) { location.hash = tab.dataset.view; }
  });
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));

  try {
    const status = await get('/api/status');
    const ready = status.models.filter((m) => m.available).length;
    setStatus(ready === status.models.length ? 'ok' : ready ? 'warn' : 'bad',
      `${ready} of ${status.models.length} models loaded`);
    predict.init(status);
    batch.init(status);
  } catch (err) {
    setStatus('bad', 'Server not reachable');
  }
  show(location.hash.slice(1));
}

boot();
