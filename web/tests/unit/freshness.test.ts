// Server-rendered freshness context: stale vs current must be stated in the HTML itself
// (readable without JavaScript), and missing values are "not recorded", never blank.
import { describe, expect, it } from 'vitest';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import Freshness from '../../src/components/Freshness.astro';
import type { Freshness as FreshnessData } from '../../src/lib/contracts';

const base: FreshnessData = {
  coverage_through: 'Round 5', dataset_status: 'verified', generated_at: '2026-09-25T00:00:00Z',
  latest_completed_match_at: '2026-09-12T09:40:00Z', latest_completed_match_date: '2026-09-12', published_at: null,
  season_active: true, source_checked_at: '2026-09-24T22:00:00Z', stale: false, stale_reason: null, validation_state: 'PASS',
};

async function render(f: FreshnessData) {
  const c = await AstroContainer.create();
  return c.renderToString(Freshness, { props: { freshness: f, season: 2026, demo: false, releaseId: 'r1' } });
}

describe('Freshness', () => {
  it('labels a current release and shows no stale banner', async () => {
    const html = await render(base);
    expect(html).toContain('Current');
    expect(html).not.toContain('data-testid="stale-banner"');
    expect(html).toContain('not published (local build)');
  });
  it('labels a stale release with its reason in server-rendered HTML', async () => {
    const html = await render({ ...base, stale: true, stale_reason: 'Source not checked for 9 days.' });
    expect(html).toContain('data-testid="stale-banner"');
    expect(html).toContain('Data may be out of date.');
    expect(html).toContain('Source not checked for 9 days.');
  });
  it('says not recorded when coverage is unknown', async () => {
    const html = await render({ ...base, coverage_through: null });
    expect(html).toMatch(/Coverage through<\/dt><dd[^>]*><span class="missing"[^>]*>not recorded/);
  });
});
