/** Public ID / resource-path helpers. Everything fetched is validated here first. */

/** Canonical resource keys are `k.` plus unpadded base64url, at most 200 bytes. */
export const SAFE_KEY_RE = /^[A-Za-z0-9][A-Za-z0-9_\-.]{0,199}$/;
/** Public ids as emitted by the Python side (domain.schemas.SAFE_ID_RE). */
export const SAFE_ID_RE = /^[A-Za-z0-9][A-Za-z0-9:_\-.]{0,159}$/;
const KEY_PREFIX = 'k.';
const MAX_KEY_BYTES = 200;

export function isSafeKey(key: string): boolean {
  return SAFE_KEY_RE.test(key) && !key.includes('..');
}

export function isSafeId(id: string): boolean {
  return SAFE_ID_RE.test(id) && !id.includes('..');
}

function asciiBytes(text: string): Uint8Array {
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c > 127) throw new Error('public id must be ASCII');
    bytes[i] = c;
  }
  return bytes;
}

function encodeB64Url(text: string): string {
  const bytes = asciiBytes(text);
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}

function decodeB64Url(token: string): string {
  if (!/^[A-Za-z0-9_-]+$/.test(token)) throw new Error('malformed public key');
  const pad = token + '='.repeat((4 - (token.length % 4)) % 4);
  const bin = atob(pad.replaceAll('-', '+').replaceAll('_', '/'));
  let text = '';
  for (let i = 0; i < bin.length; i++) {
    const c = bin.charCodeAt(i);
    if (c > 127) throw new Error('malformed public key');
    text += String.fromCharCode(c);
  }
  return text;
}

export function encodeId(id: string): string {
  if (!id) throw new Error('empty public id');
  const key = KEY_PREFIX + encodeB64Url(id);
  if (key.length > MAX_KEY_BYTES) throw new Error('encoded key exceeds 200 bytes; persist an explicit alias, do not truncate');
  return key;
}

export function decodeKey(key: string): string {
  if (key.startsWith(KEY_PREFIX)) {
    const text = decodeB64Url(key.slice(KEY_PREFIX.length));
    if (encodeId(text) !== key) throw new Error('non-canonical public key');
    return text;
  }
  // Legacy alias from the historical colon-to-`__` paths. Not used for new files.
  return key.replaceAll('__', ':');
}

/** Accept a public id, its canonical key, or a legacy `__` alias. The returned key is canonical. */
export function parsePlayerIdParam(raw: string | null | undefined): { id: string; key: string } | null {
  if (!raw) return null;
  const value = raw.trim();
  try {
    const id = value.includes(':') || value.startsWith(KEY_PREFIX) ? (value.startsWith(KEY_PREFIX) ? decodeKey(value) : value) : decodeKey(value);
    if (!isSafeId(id)) return null;
    const key = encodeId(id);
    return isSafeKey(key) ? { id, key } : null;
  } catch {
    return null;
  }
}

const SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9_\-.]*$/;

/** Relative, same-origin, contained path under the release root. */
export function isSafeResourcePath(path: string): boolean {
  if (!path || path.length > 512) return false;
  if (path.startsWith('/') || path.includes('\\') || path.includes('://') || /[?#%\s]/.test(path)) return false;
  return path.split('/').every((seg) => SEGMENT_RE.test(seg) && seg !== '.' && seg !== '..' && !seg.includes('..'));
}

export function normalizeBase(base: string): string {
  const b = base.startsWith('/') ? base : `/${base}`;
  return b.endsWith('/') ? b : `${b}/`;
}

export function releaseUrl(base: string, releaseId: string, path: string): string {
  if (!isSafeKey(releaseId)) throw new Error(`unsafe release id: ${releaseId}`);
  if (!isSafeResourcePath(path)) throw new Error(`unsafe resource path: ${path}`);
  return `${normalizeBase(base)}data/${releaseId}/${path}`;
}

/** Join a site-relative route onto the configured base ("/" or "/SuperCoach-VIA/"). */
export function withBase(base: string, route: string): string {
  const r = route.replace(/^\/+/, '');
  return `${normalizeBase(base)}${r}`;
}
