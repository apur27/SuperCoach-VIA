import { readFileSync } from 'node:fs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import DataStatus from '../../src/pages/data-status/index.astro';
import { manifest, overview } from '../../src/lib/release-server';
import { CORRECTED_SNAPSHOT_ID } from '../../src/lib/provisional';

vi.mock('../../src/lib/release-server', () => {
  const fixture = (path: string) => JSON.parse(readFileSync(new URL(`../fixtures/demo-release/${path}`, import.meta.url), 'utf8'));
  return {
    manifest: vi.fn(() => fixture('release.json')),
    overview: vi.fn(() => fixture('overview.json')),
    readResource: (_kind: string, path: string) => fixture(path),
  };
});

beforeEach(() => {
  vi.mocked(manifest).mockReturnValue({ ...manifest(), snapshot_id: CORRECTED_SNAPSHOT_ID, demo: false });
});

describe('source confidence and dataset freshness', () => {
  it.each([false, true])('keeps freshness visible beside the UNKNOWN audit (stale=%s)', async (stale) => {
    const ov = overview();
    vi.mocked(overview).mockReturnValue({ ...ov, freshness: { ...ov.freshness, stale, stale_reason: 'Source last checked earlier.' } });
    const c = await AstroContainer.create();
    const html = await c.renderToString(DataStatus);
    expect(html).toContain('AFL Tables source audit: UNKNOWN');
    expect(html).toContain('63 unresolved source cells per layer');
    expect(html).toContain('content="noindex"');
    expect(html).toContain(stale ? '<strong>Stale:</strong>' : 'Dataset freshness: current as of the last source check.');
    expect(html).not.toContain('AFL Tables source audit: FAIL');
  });
});
