import { describe, expect, it } from 'vitest';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';
import ProvisionalStatus from '../../src/components/ProvisionalStatus.astro';
import ProvisionalRobots from '../../src/components/ProvisionalRobots.astro';
import ProvisionalAudit from '../../src/components/ProvisionalAudit.astro';
import { provisionalAudit, PROVISIONAL_SNAPSHOT_ID } from '../../src/lib/provisional';
const props = { snapshotId: 'sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0', demo: false, generatedAt: '2026-09-29T21:39:00Z', base: '/SuperCoach-VIA/' };
describe('pinned provisional audit', () => {
  it('matches only the known real snapshot', () => {
    expect(provisionalAudit(PROVISIONAL_SNAPSHOT_ID, false)).not.toBeNull();
    expect(provisionalAudit('sha256:another', false)).toBeNull();
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
  it.each([{ ...props, snapshotId: 'sha256:another' }, { ...props, demo: true }])('does not apply this audit to another snapshot or DEMO', async (input) => {
    const c = await AstroContainer.create();
    for (const component of [ProvisionalStatus, ProvisionalRobots, ProvisionalAudit]) expect((await c.renderToString(component, { props: input })).trim()).toBe('');
  });
});
