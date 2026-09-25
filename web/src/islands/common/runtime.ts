/** Client runtime for islands: one release, one loader, manifest-verified. */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ResourceKind, ResourceTypes } from '../../lib/contracts';
import { createLoader, FetchFailedError, InvalidPayloadError, NotFoundError, ReleaseUnavailableError, type Loader } from '../../lib/data';
import { announceReleaseUnavailable } from '../../lib/prefs';

let loaderPromise: Promise<Loader> | null = null;

function pageRelease() {
  const el = document.documentElement;
  return { releaseId: el.dataset.release ?? '', base: el.dataset.base ?? '/' };
}

/** Loads release.json once per page and checks it is the release the HTML was built for. */
export function getLoader(): Promise<Loader> {
  loaderPromise ??= (async () => {
    const { releaseId, base } = pageRelease();
    const bootstrap = createLoader({ base, releaseId });
    const manifest = await bootstrap.load('release', 'release.json');
    if (manifest.release_id !== releaseId) throw new ReleaseUnavailableError('release manifest id mismatch');
    return createLoader({ base, releaseId, manifest });
  })();
  loaderPromise.catch(() => {
    loaderPromise = null;
  });
  return loaderPromise;
}

export type LoadState<T> =
  | { status: 'idle' }
  | { status: 'loading'; previous?: T }
  | { status: 'success'; data: T }
  | { status: 'error'; kind: ErrorKind; message: string; previous?: T };
export type ErrorKind = 'release' | 'invalid' | 'network' | 'notfound';

export function classifyError(e: unknown): { kind: ErrorKind; message: string } {
  if (e instanceof NotFoundError) return { kind: 'notfound', message: e.message };
  if (e instanceof ReleaseUnavailableError) return { kind: 'release', message: e.message };
  if (e instanceof InvalidPayloadError) return { kind: 'invalid', message: e.message };
  if (e instanceof FetchFailedError) return { kind: 'network', message: e.message };
  return { kind: 'network', message: e instanceof Error ? e.message : 'Unknown error' };
}

/**
 * Fetch one validated resource. `path === null` means "nothing to load yet" (idle).
 * Obsolete requests are aborted; the last good value is kept while revalidating.
 */
export function useResource<K extends ResourceKind>(kind: K, path: string | null) {
  const [state, setState] = useState<LoadState<ResourceTypes[K]>>({ status: 'idle' }); // SSR/no-JS never shows a spinner
  const [attempt, setAttempt] = useState(0);
  const last = useRef<ResourceTypes[K] | undefined>(undefined);
  useEffect(() => {
    if (!path) {
      setState({ status: 'idle' });
      return undefined;
    }
    const ac = new AbortController();
    setState(last.current !== undefined ? { status: 'loading', previous: last.current } : { status: 'loading' });
    getLoader()
      .then((l) => l.load(kind, path, { signal: ac.signal }))
      .then((data) => {
        if (ac.signal.aborted) return;
        last.current = data;
        setState({ status: 'success', data });
      })
      .catch((e: unknown) => {
        if (ac.signal.aborted || (e as Error)?.name === 'AbortError') return;
        const c = classifyError(e);
        if (c.kind === 'release') announceReleaseUnavailable();
        setState(last.current !== undefined ? { status: 'error', ...c, previous: last.current } : { status: 'error', ...c });
      });
    return () => ac.abort();
  }, [kind, path, attempt]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { state, retry };
}

/** URL-as-source-of-truth: read on mount and on back/forward; write with push/replace. */
export function useUrlSearch(): [string, (search: string, mode?: 'push' | 'replace') => void] {
  const [search, setSearch] = useState(() => (typeof window === 'undefined' ? '' : window.location.search));
  useEffect(() => {
    const onPop = () => setSearch(window.location.search);
    window.addEventListener('popstate', onPop);
    setSearch(window.location.search);
    return () => window.removeEventListener('popstate', onPop);
  }, []);
  const update = useCallback((next: string, mode: 'push' | 'replace' = 'push') => {
    const url = `${window.location.pathname}${next}${window.location.hash}`;
    if (next === window.location.search) return;
    if (mode === 'push') window.history.pushState(null, '', url);
    else window.history.replaceState(null, '', url);
    setSearch(next);
  }, []);
  return [search, update];
}

export function useHydrated(): boolean {
  const [h, setH] = useState(false);
  useEffect(() => setH(true), []);
  return h;
}

export function siteBase(): string {
  return typeof document === 'undefined' ? '/' : (document.documentElement.dataset.base ?? '/');
}

/** Detail shells render a generic H1 server-side; islands refine it once data loads. */
export function useHeading(text: string | null) {
  useEffect(() => {
    if (!text) return;
    const h1 = document.getElementById('page-h1');
    if (h1) h1.textContent = text;
    const demo = document.documentElement.dataset.demo === 'true' ? 'DEMO · ' : '';
    document.title = `${demo}${text} · SuperCoach VIA`;
  }, [text]);
}
