import { describe, expect, it } from 'vitest';
import { liveFreshness, POLL_MS } from '../../src/lib/live';

describe('live freshness', () => {
  const now = Date.parse('2026-09-25T02:00:00Z');
  it('records the collector interval without polling an archived release', () => expect(POLL_MS).toBe(90_000));
  it('final snapshots are final and never polled', () => {
    expect(liveFreshness({ final: true, fetched_at: '2026-09-12T12:30:00Z' }, now)).toEqual({ state: 'final', poll: false });
  });
  it('a recent captured snapshot is labelled as a snapshot and not polled', () => {
    expect(liveFreshness({ final: false, fetched_at: '2026-09-25T01:58:00Z' }, now)).toEqual({ state: 'snapshot', poll: false });
  });
  it('old in-progress snapshots are stale and not polled', () => {
    expect(liveFreshness({ final: false, fetched_at: '2026-09-24T23:50:00Z' }, now)).toEqual({ state: 'stale', poll: false });
  });
  it('unknown fetch time is stale', () => {
    expect(liveFreshness({ final: false, fetched_at: null }, now)).toEqual({ state: 'stale', poll: false });
  });
});
