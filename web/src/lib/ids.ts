/** Public ID / resource-path helpers. Everything fetched is validated here first. */

/** Resource keys: encoded public ids (':' -> '__'). Mirrors the release key grammar. */
export const SAFE_KEY_RE = /^[A-Za-z0-9][A-Za-z0-9_\-.]{0,159}$/;
/** Public ids as emitted by the Python side (domain.schemas.SAFE_ID_RE). */
export const SAFE_ID_RE = /^[A-Za-z0-9][A-Za-z0-9:_\-.]{0,159}$/;

export function isSafeKey(key: string): boolean {
  return SAFE_KEY_RE.test(key) && !key.includes('..');
}

export function isSafeId(id: string): boolean {
  return SAFE_ID_RE.test(id) && !id.includes('..');
}

export function encodeId(id: string): string {
  return id.replaceAll(':', '__');
}

export function decodeKey(key: string): string {
  return key.replaceAll('__', ':');
}

/** Accept either a public id or its encoded key from a query parameter. */
export function parsePlayerIdParam(raw: string | null | undefined): { id: string; key: string } | null {
  if (!raw) return null;
  const value = raw.trim();
  if (value.includes(':')) {
    if (!isSafeId(value)) return null;
    const key = encodeId(value);
    return isSafeKey(key) ? { id: value, key } : null;
  }
  if (!isSafeKey(value)) return null;
  return { id: decodeKey(value), key: value };
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
