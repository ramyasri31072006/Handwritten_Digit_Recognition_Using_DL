import { get, post } from './api.js';
import { $, $$, h, clear, drawDigit, pct, num } from './ui.js';
import { hbars } from './charts.js';

const PAGE_SIZE = 24;
const state = { token: 0, samplePage: 0, uploadUrl: null };
let pad, ctx, drawing = false, inked = false, timer = null;

export function init(status) {
  initModes();
  initPad();
  initUpload();
  initSamples();
  $('#probModel').addEventListener('change', drawProbabilities);
  if (!status.models.some((m) => m.available)) {
    showError('No trained models were found. Copy your model files into the models folder and restart the server (see the README).');
  }
}

// ---- input modes -------------------------------------------------------------------------
function initModes() {
  $('#inputMode').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-mode]');
    if (button) setMode(button.dataset.mode);
  });
}

function setMode(mode) {
  $$('#inputMode button').forEach((b) => b.classList.toggle('on', b.dataset.mode === mode));
  ['draw', 'upload', 'sample'].forEach((m) => $(`#mode-${m}`).classList.toggle('hidden', m !== mode));
  if (mode === 'sample' && !$('#sampleGrid').children.length) loadSamples(0);
}

// ---- drawing pad -------------------------------------------------------------------------
function initPad() {
  pad = $('#pad');
  ctx = pad.getContext('2d');
  resetPad();

  const position = (e) => {
    const box = pad.getBoundingClientRect();
    return [(e.clientX - box.left) * (pad.width / box.width), (e.clientY - box.top) * (pad.height / box.height)];
  };
  pad.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    clearTimeout(timer);
    drawing = inked = true;
    const [x, y] = position(e);
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x, y);
  });
  pad.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const [x, y] = position(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  });
  const finish = () => {
    if (!drawing) return;
    drawing = false;
    ctx.beginPath();
    timer = setTimeout(predictDrawing, 300);
  };
  pad.addEventListener('pointerup', finish);
  pad.addEventListener('pointercancel', finish);

  $('#brush').addEventListener('input', (e) => { ctx.lineWidth = Number(e.target.value); });
  $('#btnClear').addEventListener('click', () => { resetPad(); resetResults(); });
  $('#btnPredict').addEventListener('click', predictDrawing);
}

function resetPad() {
  ctx.fillStyle = '#0d1226';
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.fillStyle = ctx.strokeStyle = '#ffffff';
  ctx.lineCap = ctx.lineJoin = 'round';
  ctx.lineWidth = Number($('#brush').value);
  inked = false;
}

function predictDrawing() {
  if (!inked) return showError('Draw a digit on the slate first.');
  predict('/api/predict/image', { image: pad.toDataURL('image/png') });
}

// ---- image upload ------------------------------------------------------------------------
function initUpload() {
  const input = $('#imageFile'), zone = $('#dropzone');
  input.addEventListener('change', () => input.files[0] && handleFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
  $('#polarity').addEventListener('change', () => state.uploadUrl && sendUpload());
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) return showError('Choose an image file (PNG, JPG, GIF, WEBP or BMP).');
  if (file.size > 15 * 1024 * 1024) return showError('That image is larger than 15 MB. Use a smaller one.');
  const reader = new FileReader();
  reader.onerror = () => showError('The file could not be read.');
  reader.onload = () => {
    state.uploadUrl = reader.result;
    clear($('#uploadPreview')).append(h('img', { src: reader.result, alt: 'Uploaded digit' }), h('span', { class: 'kicker' }, file.name));
    sendUpload();
  };
  reader.readAsDataURL(file);
}

function sendUpload() {
  const mode = $('#polarity').value;
  predict('/api/predict/image', { image: state.uploadUrl, invert: mode === 'auto' ? null : mode === 'light' });
}

// ---- dataset samples ---------------------------------------------------------------------
function initSamples() {
  $('#samplePrev').addEventListener('click', () => loadSamples(state.samplePage - 1));
  $('#sampleNext').addEventListener('click', () => loadSamples(state.samplePage + 1));
}

async function loadSamples(page) {
  const grid = $('#sampleGrid');
  try {
    const res = await get(`/api/samples?offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}`);
    state.samplePage = page;
    clear(grid);
    for (const sample of res.samples) {
      const canvas = h('canvas', {});
      drawDigit(canvas, sample.pixels);
      const button = h('button', {
        title: `test.csv row ${sample.index + 1}`, 'aria-label': `Sample from row ${sample.index + 1}`,
        onclick: () => {
          $$('#sampleGrid button').forEach((b) => b.classList.remove('sel'));
          button.classList.add('sel');
          predict('/api/predict/pixels', { pixels: sample.pixels });
        },
      }, canvas);
      grid.append(button);
    }
    $('#samplePage').textContent = `Rows ${num(res.offset + 1)} to ${num(res.offset + res.samples.length)} of ${num(res.total)}`;
    $('#samplePrev').disabled = page === 0;
    $('#sampleNext').disabled = (page + 1) * PAGE_SIZE >= res.total;
  } catch (err) {
    clear(grid).append(h('div', { class: 'notice error', style: 'grid-column:1/-1' }, err.message));
    $('#samplePrev').disabled = $('#sampleNext').disabled = true;
  }
}

// ---- prediction + rendering --------------------------------------------------------------
async function predict(endpoint, payload) {
  const token = ++state.token; // ignore out-of-date responses
  const panel = $('#resultPanel');
  panel.classList.add('loading');
  try {
    const res = await post(endpoint, payload);
    if (token === state.token) showResult(res);
  } catch (err) {
    if (token === state.token) showError(err.message);
  } finally {
    if (token === state.token) panel.classList.remove('loading');
  }
}

function showError(message) {
  $('#resultEmpty').classList.add('hidden');
  $('#resultBody').classList.add('hidden');
  const box = $('#resultError');
  box.textContent = message;
  box.classList.remove('hidden');
}

function resetResults() {
  state.token++;
  $('#resultPanel').classList.remove('loading');
  $('#resultBody').classList.add('hidden');
  $('#resultError').classList.add('hidden');
  $('#resultEmpty').classList.remove('hidden');
}

let lastResults = [];

function showResult(res) {
  $('#resultEmpty').classList.add('hidden');
  $('#resultError').classList.add('hidden');
  $('#resultBody').classList.remove('hidden');
  lastResults = res.results;

  const c = res.consensus;
  $('#bigDigit').textContent = c ? c.digit : '?';
  $('#agreeText').textContent = !c ? 'No model could predict this image'
    : c.of === 1 ? 'One model loaded'
    : c.votes === c.of ? `All ${c.of} models agree`
    : `${c.votes} of ${c.of} models predict ${c.digit}`;
  drawDigit($('#seen'), res.processed);

  const rows = clear($('#modelRows'));
  for (const r of res.results) {
    const odd = c && r.prediction !== undefined && r.prediction !== c.digit;
    rows.append(h('div', { class: `model-row m-${r.model} ${odd ? 'odd' : ''}` },
      h('div', { class: 'name' }, h('i', { class: 'swatch' }), r.label),
      r.error ? h('div', { class: 'err' }, r.error) : [
        h('div', { class: 'digit' }, r.prediction),
        h('div', { class: 'conf' }, r.confidence != null
          ? [h('div', { class: 'bar' }, h('span', { style: `width:${(r.confidence * 100).toFixed(1)}%` })), h('b', {}, pct(r.confidence, 1))]
          : h('span', { class: 'none' }, 'Label only, no probabilities')),
        h('div', { class: 'lat' }, `${r.latency_ms} ms`),
      ]));
  }

  const select = clear($('#probModel'));
  const withProbs = res.results.filter((r) => r.probabilities);
  withProbs.forEach((r) => select.append(h('option', { value: r.model }, r.label)));
  select.disabled = !withProbs.length;
  drawProbabilities();
}

function drawProbabilities() {
  const box = $('#probChart');
  const chosen = lastResults.find((r) => r.model === $('#probModel').value && r.probabilities);
  if (!chosen) {
    clear(box).append(h('p', { class: 'hint' }, 'None of the loaded models returns probabilities.'));
    return;
  }
  hbars(box, chosen.probabilities.map((p, digit) => ({
    label: digit, frac: p, text: pct(p, 1), model: chosen.model,
    top: digit === chosen.prediction, dim: digit !== chosen.prediction,
  })));
}
