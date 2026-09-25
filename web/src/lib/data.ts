/** Release-scoped, same-origin, validated data loading for islands. */
import type { ReleaseManifest, ResourceKind, ResourceTypes } from './contracts';
import { validateAsync } from './validate-client';
import { releaseUrl } from './ids';

/** The resource no longer exists for this release: the tab is older than the deployed site. */
export class ReleaseUnavailableError extends Error {
  override name = 'ReleaseUnavailableError';
}
/** The release is still served but this resource does not exist (e.g. unknown player id). */
export class NotFoundError extends Error {
  override name = 'NotFoundError';
}
/** Payload is not JSON, fails the schema, or its hash disagrees with the manifest. */
export class InvalidPayloadError extends Error {
  override name = 'InvalidPayloadError';
}
/** Network/5xx failure; retrying may help. */
export class FetchFailedError extends Error {
  override name = 'FetchFailedError';
}

export interface LoaderOptions {
  base: string;
  releaseId: string;
  manifest?: ReleaseManifest | null;
  fetchImpl?: (url: string, init?: RequestInit) => Promise<Response>;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null; // non-secure context: schema validation still applies
  const digest = await subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export function createLoader(opts: LoaderOptions) {
  const fetchImpl = opts.fetchImpl ?? ((u: string, i?: RequestInit) => fetch(u, i));
  const cache = new Map<string, unknown>();
  const expected = new Map<string, { sha256: string; bytes: number }>();
  for (const ref of Object.values(opts.manifest?.resources ?? {})) expected.set(ref.path, ref);

  async function load<K extends ResourceKind>(kind: K, path: string, init: { signal?: AbortSignal; fresh?: boolean } = {}): Promise<ResourceTypes[K]> {
    let url: string;
    try {
      url = releaseUrl(opts.base, opts.releaseId, path);
    } catch (e) {
      throw new Error(`unsafe resource path refused: ${(e as Error).message}`);
    }
    const key = `${kind}:${path}`;
    if (!init.fresh && cache.has(key)) return cache.get(key) as ResourceTypes[K];
    if (init.signal?.aborted) throw new DOMException('aborted', 'AbortError');
    let res: Response;
    try {
      res = await fetchImpl(url, { credentials: 'same-origin', ...(init.fresh ? { cache: 'no-store' as RequestCache } : {}), ...(init.signal ? { signal: init.signal } : {}) });
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e;
      throw new FetchFailedError(`network error loading ${path}`);
    }
    if (res.status === 404 || res.status === 410) {
      // Distinguish "no such item" from "this whole release is gone" (old tab after a deploy).
      if (path !== 'release.json') {
        const probe = await fetchImpl(releaseUrl(opts.base, opts.releaseId, 'release.json'), { credentials: 'same-origin', cache: 'no-store' }).catch(() => null);
        if (probe?.ok) throw new NotFoundError(`${path} does not exist in release ${opts.releaseId}`);
      }
      throw new ReleaseUnavailableError(`${path} is not available for release ${opts.releaseId}`);
    }
    if (!res.ok) throw new FetchFailedError(`HTTP ${res.status} loading ${path}`);
    const bytes = await res.arrayBuffer();
    const exp = expected.get(path);
    if (exp) {
      const digest = await sha256Hex(bytes);
      if (bytes.byteLength !== exp.bytes || (digest !== null && digest !== exp.sha256)) {
        throw new InvalidPayloadError(`${path} does not match the release manifest checksum`);
      }
    }
    let data: unknown;
    try {
      data = JSON.parse(new TextDecoder().decode(bytes));
    } catch {
      throw new InvalidPayloadError(`${path} is not valid JSON`);
    }
    const result = await validateAsync(kind, data);
    if (!result.ok) throw new InvalidPayloadError(`${path} failed validation: ${result.error}`);
    cache.set(key, result.value);
    return result.value;
  }
  return { load, releaseId: opts.releaseId, base: opts.base };
}

export type Loader = ReturnType<typeof createLoader>;
