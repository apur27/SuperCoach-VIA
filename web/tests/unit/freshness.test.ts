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
    expect(html).toContain('not recorded in release');
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
  it('keeps the strip compact: secondary provenance sits in a disclosure (O55-08)', async () => {
    const html = await render(base);
    const fold = html.indexOf('<details');
    expect(fold).toBeGreaterThan(-1);
    expect(html).toContain('Release details');
    for (const label of ['Season', 'Coverage through', 'Source checked', 'Status', 'Source &amp; method']) {
      expect(html.indexOf(`>${label}</dt>`), label).toBeGreaterThan(-1);
      expect(html.indexOf(`>${label}</dt>`), label).toBeLessThan(fold);
    }
    for (const label of ['Generated', 'Published', 'Release']) expect(html.indexOf(`>${label}</dt>`), label).toBeGreaterThan(fold);
  });
  it('never hides a stale warning inside the disclosure', async () => {
    const html = await render({ ...base, stale: true, stale_reason: 'Source not checked for 9 days.' });
    expect(html.indexOf('data-testid="stale-banner"')).toBeGreaterThan(html.indexOf('</details>'));
  });
});

