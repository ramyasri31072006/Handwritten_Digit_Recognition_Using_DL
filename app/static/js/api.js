// Thin fetch wrapper: always resolves to parsed JSON or throws an Error with a readable message.
export async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error('Cannot reach the server. Is app.py still running?');
  }
  let body = null;
  try { body = await response.json(); } catch { /* non-JSON body */ }
  if (!response.ok) throw new Error((body && body.error) || `Request failed (${response.status}).`);
  return body;
}

export const get = (path) => api(path);
export const post = (path, payload) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
