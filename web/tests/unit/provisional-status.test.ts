import { describe, expect, it } from 'vitest';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import ProvisionalStatus from '../../src/components/ProvisionalStatus.astro';
import ProvisionalRobots from '../../src/components/ProvisionalRobots.astro';
import ProvisionalAudit from '../../src/components/ProvisionalAudit.astro';
import { provisionalAudit, PROVISIONAL_SNAPSHOT_ID, CORRECTED_SNAPSHOT_ID } from '../../src/lib/provisional';
const props = { snapshotId: 'sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0', demo: false, generatedAt: '2026-09-29T21:39:00Z', base: '/SuperCoach-VIA/' };
describe('pinned provisional audit', () => {
  it('binds the known report only to its real snapshot', () => {
    expect(provisionalAudit(PROVISIONAL_SNAPSHOT_ID, false)).not.toBeNull();
    expect(provisionalAudit('sha256:another', false)?.verdict).toBe('UNAUDITED');
    expect(provisionalAudit(PROVISIONAL_SNAPSHOT_ID, true)).toBeNull();
  });
  it('renders the warning, generation date and base-aware detail link without JavaScript', async () => {
    const c = await AstroContainer.create();
    const html = await c.renderToString(ProvisionalStatus, { props });
    expect(html).toContain('Provisional data — known audit failures');
    expect(html).toContain('Snapshot created');
    expect(html).toContain('2026-09-29T20:38:16.571555Z');
    expect(html).toContain('Build generated');
    expect(html).toContain('2026-09-29');
    expect(html).toContain('href="/SuperCoach-VIA/data-status/"');
    expect(html).not.toContain('<script');
    expect(await c.renderToString(ProvisionalRobots, { props })).toContain('content="noindex"');
  });
  it('shows audit failures separately from release consistency and links public evidence', async () => {
    const c = await AstroContainer.create();
    const html = await c.renderToString(ProvisionalAudit, { props });
    for (const text of ['58eff517a29d30b932087f919a31d36567e8be330301ecff53feb06494d8db28', 'AFL Tables source audit: FAIL', 'Release consistency: PASS', 'missing appearances', 'zero', 'historical Brownlow', 'unresolved evidence', 'no_valid_future_fixture', props.snapshotId, 'README.md#data-status', 'docs/hall-of-fame/provisional/README.md']) expect(html).toContain(text);
  });
  it('keeps an unpinned real snapshot provisional without borrowing the FAIL report', async () => {
    const c = await AstroContainer.create();
    const input = { ...props, snapshotId: 'sha256:another' };
    expect(await c.renderToString(ProvisionalStatus, { props: input })).toContain('source audit unavailable');
    expect(await c.renderToString(ProvisionalRobots, { props: input })).toContain('content="noindex"');
    const html = await c.renderToString(ProvisionalAudit, { props: input });
    expect(html).toContain('AFL Tables source audit: UNAUDITED');
    expect(html).toContain(input.snapshotId);
    expect(html).not.toContain('58eff517');
    expect(html).not.toContain('Release consistency: PASS');
    expect(html).not.toContain('2026-09-29T20:38:16.571555Z');
  });
  it('binds the corrected UNKNOWN status to the parent full audit and candidate 2026 audit', async () => {
    const c = await AstroContainer.create();
    const input = { ...props, snapshotId: CORRECTED_SNAPSHOT_ID };
    const metadata = provisionalAudit(CORRECTED_SNAPSHOT_ID, false);
    expect(metadata?.verdict).toBe('UNKNOWN');
    const html = await c.renderToString(ProvisionalAudit, { props: input });
    for (const text of ['AFL Tables source audit: UNKNOWN', 'zero confirmed discrepancies', '63 unresolved source cells per layer', 'f1abd8c2b7f8d6b3812e73f91a4403add1844f3ad9037139e7dc373a0ed91703', '81ac70ba8aa25110119a91779d2c2df91088b2f0f87408db3546de8109a4509c', '6fecac8e1c8eb804774eb70a5f9d03bd5448ee86d43d1253472ca0f1ea7d9c2a', '2026 only', 'Source audit run record', 'afltables-candidate-source-coverage-20261007.json']) expect(html).toContain(text);
    expect(html).not.toContain('58eff517');
    expect(html).not.toContain('Release consistency: PASS');
    expect(await c.renderToString(ProvisionalRobots, { props: input })).toContain('content="noindex"');
    const banner = await c.renderToString(ProvisionalStatus, { props: input });
    expect(banner).toContain('source audit UNKNOWN');
    expect(banner).toContain('2026-10-05T20:43:39.205629Z');
  });
  it('does not apply source audit metadata to DEMO' , async () => {
    const input = { ...props, demo: true };
    const c = await AstroContainer.create();
    for (const component of [ProvisionalStatus, ProvisionalRobots, ProvisionalAudit]) expect((await c.renderToString(component, { props: input })).trim()).toBe('');
  });
});
