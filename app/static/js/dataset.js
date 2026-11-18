import { get } from './api.js';
import { $, h, clear, drawDigit, notice, num } from './ui.js';
import { hbars } from './charts.js';

let loaded = false;

export async function onShow() {
  if (loaded) return;
  const out = $('#dataOut');
  try {
    const info = await get('/api/dataset');
    loaded = true;
    render(clear(out), info);
  } catch (err) {
    clear(out).append(notice(err.message));
  }
}

const fact = (label, value, small) => h('div', {}, h('dt', {}, label), h('dd', {}, value, small ? h('small', {}, ` ${small}`) : null));

function render(out, info) {
  out.append(h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'The data')),
    h('p', { class: 'panel-note' }, 'Each row is one 28 x 28 grayscale image of a handwritten digit, flattened into 784 pixel columns (pixel0 to pixel783) with values from 0 (black) to 255 (white). train.csv adds a label column with the digit, 0 to 9.'),
    h('dl', { class: 'facts' },
      fact('train.csv', '42,000 x 785', 'from your notebooks'),
      fact('Training split', '33,600', 'images (80%)'),
      fact('Held-out split', '8,400', 'images (20%)'),
      info.test_csv ? fact('test.csv', `${num(info.test_csv.rows)} x ${num(info.test_csv.columns)}`, 'no labels') : null)));

  out.append(h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'How the models were prepared')),
    h('p', { class: 'panel-note' }, 'The same steps run in notebooks 02 to 05. The CNN additionally reshapes each image to 28 x 28 x 1.'),
    h('ol', { class: 'flow' },
      step('Load train.csv', 'label in column 0'),
      step('Split label and pixels', 'X = pixels, y = digit'),
      step('Scale pixels', 'divide by 255, range 0 to 1'),
      step('Stratified split', '80% train, 20% test, random_state=42'),
      step('Train', 'MLP, CNN, SVM, kNN'),
      step('Evaluate', 'accuracy, precision, recall, F1'),
      step('Save', '.pkl for sklearn, .keras for CNN'))));

  if (info.test_csv) {
    const canvas = h('canvas', { class: 'heat', 'aria-label': 'Average test image' });
    drawDigit(canvas, info.mean_image, { normalise: true });
    const t = info.test_csv;
    out.append(h('div', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h2', {}, 'What test.csv looks like')),
      h('div', { class: 'grid even' },
        h('div', {}, canvas, h('p', { class: 'axis-note' }, 'Average of all test.csv images. Ink concentrates in the centre, which is why drawn and uploaded pictures are centred before the models see them.')),
        h('dl', { class: 'facts', style: 'align-content:start' },
          fact('Rows', num(t.rows)), fact('Pixel range', `${t.min} to ${t.max}`),
          fact('Mean pixel value', t.mean_intensity), fact('Blank images', `${(t.blank_share * 100).toFixed(2)}%`)))));
  } else {
    out.append(notice('test.csv was not found in the data folder, so live dataset statistics and the sample browser are unavailable.', 'warn'));
  }

  if (info.class_counts) {
    const total = info.class_counts.reduce((a, b) => a + b, 0), max = Math.max(...info.class_counts);
    const box = h('div', {});
    out.append(h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Digits in train.csv')), box));
    hbars(box, info.class_counts.map((c, d) => ({ label: d, frac: c / max, text: `${num(c)} (${((c / total) * 100).toFixed(1)}%)`, model: 'cnn' })));
  }
}

const step = (title, detail) => h('li', {}, h('b', {}, title), ' ', h('span', {}, detail));
