import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createLoader, ReleaseUnavailableError, InvalidPayloadError, FetchFailedError, NotFoundError } from '../../src/lib/data';

const FIX = resolve(__dirname, '../fixtures/demo-release');
const read = (rel: string) => readFileSync(resolve(FIX, rel), 'utf8');

function fakeFetch(map: Record<string, { status: number; body: string }>) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.signal?.aborted) throw new DOMException('aborted', 'AbortError');
    const hit = map[url];
    if (!hit) return new Response('not found', { status: 404 });
    return new Response(hit.body, { status: hit.status, headers: { 'content-type': 'application/json' } });
  });
}

describe('release loader', () => {
  const base = '/SuperCoach-VIA/';
  const rid = '20260925T000000Z-demo';
  const url = (p: string) => `${base}data/${rid}/${p}`;

  it('fetches, validates and types a resource', async () => {
    const fetchImpl = fakeFetch({ [url('overview.json')]: { status: 200, body: read('overview.json') } });
    const loader = createLoader({ base, releaseId: rid, fetchImpl });
    const ov = await loader.load('overview', 'overview.json');
    expect(ov.release_id).toBe(rid);
    expect(fetchImpl).toHaveBeenCalledWith(url('overview.json'), expect.objectContaining({ credentials: 'same-origin' }));
  });
  it('treats 404 as a missing release (newer release available)', async () => {
    const loader = createLoader({ base, releaseId: rid, fetchImpl: fakeFetch({}) });
    await expect(loader.load('overview', 'overview.json')).rejects.toBeInstanceOf(ReleaseUnavailableError);
  });
  it('treats 404 as not-found when the release itself is still served', async () => {
    const loader = createLoader({ base, releaseId: rid, fetchImpl: fakeFetch({ [url('release.json')]: { status: 200, body: read('release.json') } }) });
    await expect(loader.load('player_detail', 'players/legacy__nobody.json')).rejects.toBeInstanceOf(NotFoundError);
  });
  it('rejects malformed payloads and schema drift', async () => {
    const bad = JSON.parse(read('overview.json'));
    bad.freshness.generated_at = null;
    const loader = createLoader({ base, releaseId: rid, fetchImpl: fakeFetch({ [url('overview.json')]: { status: 200, body: JSON.stringify(bad) }, [url('quality.json')]: { status: 200, body: '{nope' } }) });
    await expect(loader.load('overview', 'overview.json')).rejects.toBeInstanceOf(InvalidPayloadError);
    await expect(loader.load('quality', 'quality.json')).rejects.toBeInstanceOf(InvalidPayloadError);
  });
  it('reports server errors as retryable fetch failures', async () => {
    const loader = createLoader({ base, releaseId: rid, fetchImpl: fakeFetch({ [url('overview.json')]: { status: 503, body: '' } }) });
    await expect(loader.load('overview', 'overview.json')).rejects.toBeInstanceOf(FetchFailedError);
  });
  it('refuses unsafe paths without fetching', async () => {
    const fetchImpl = fakeFetch({});
    const loader = createLoader({ base, releaseId: rid, fetchImpl });
    await expect(loader.load('overview', '../other/overview.json')).rejects.toThrow(/unsafe/);
    await expect(loader.load('overview', 'https://evil.test/x.json')).rejects.toThrow(/unsafe/);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
  it('verifies sha256 for manifest-listed resources', async () => {
    const manifest = JSON.parse(read('release.json'));
    const tampered = read('overview.json').replace('DEMO release', 'DEMO releasX');
    const loader = createLoader({ base, releaseId: rid, manifest, fetchImpl: fakeFetch({ [url('overview.json')]: { status: 200, body: tampered } }) });
    await expect(loader.load('overview', 'overview.json')).rejects.toBeInstanceOf(InvalidPayloadError);
    const good = createLoader({ base, releaseId: rid, manifest, fetchImpl: fakeFetch({ [url('overview.json')]: { status: 200, body: read('overview.json') } }) });
    await expect(good.load('overview', 'overview.json')).resolves.toBeTruthy();
  });
  it('caches successful loads and propagates aborts', async () => {
    const fetchImpl = fakeFetch({ [url('quality.json')]: { status: 200, body: read('quality.json') } });
    const loader = createLoader({ base, releaseId: rid, fetchImpl });
    await loader.load('quality', 'quality.json');
    await loader.load('quality', 'quality.json');
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const ac = new AbortController();
    ac.abort();
    await expect(loader.load('overview', 'overview.json', { signal: ac.signal })).rejects.toMatchObject({ name: 'AbortError' });
  });
});

describe('fresh loads', () => {
  it('bypass the cache for polling', async () => {
    const base = '/';
    const rid = 'r1';
    const body = readFileSync(resolve(__dirname, '../fixtures/demo-release/live/demo-live-final/latest.json'), 'utf8');
    const fetchImpl = vi.fn(async () => new Response(body, { status: 200 }));
    const loader = createLoader({ base, releaseId: rid, fetchImpl });
    await loader.load('live_snapshot', 'live/demo-live-final/latest.json');
    await loader.load('live_snapshot', 'live/demo-live-final/latest.json', { fresh: true });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(fetchImpl).toHaveBeenLastCalledWith('/data/r1/live/demo-live-final/latest.json', expect.objectContaining({ cache: 'no-store' }));
  });
});
