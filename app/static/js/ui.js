// Small DOM helpers. Text is always inserted as text nodes, never as HTML.
export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) { el.replaceChildren(); return el; }
export const pct = (fraction, digits = 2) => `${(fraction * 100).toFixed(digits)}%`;
export const num = (n) => Number(n).toLocaleString();

let toastTimer = null;
export function toast(message) {
  document.querySelector('.toast')?.remove();
  document.body.append(h('div', { class: 'toast', role: 'status' }, message));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => document.querySelector('.toast')?.remove(), 3200);
}

export function notice(message, kind = 'error') {
  return h('div', { class: `notice ${kind}` }, message);
}

// Paint 784 pixel values (0-255) onto a canvas: white ink on the same ink-blue as the drawing slate.
const BASE = [13, 18, 38];
export function drawDigit(canvas, pixels, { normalise = false } = {}) {
  canvas.width = 28; canvas.height = 28;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(28, 28);
  const top = normalise ? Math.max(1, ...pixels) : 255;
  for (let i = 0; i < 784; i++) {
    const t = Math.min(1, pixels[i] / top);
    for (let c = 0; c < 3; c++) image.data[i * 4 + c] = BASE[c] + (255 - BASE[c]) * t;
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}
