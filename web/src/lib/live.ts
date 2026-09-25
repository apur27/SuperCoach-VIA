export const POLL_MS = 90_000;
/** An in-progress snapshot older than this is shown as delayed/stale. */
export const STALE_AFTER_MS = 5 * 60_000;

export function liveFreshness(s: { final: boolean; fetched_at: string | null }, now: number): { state: 'final' | 'live' | 'stale'; poll: boolean } {
  if (s.final) return { state: 'final', poll: false };
  const t = s.fetched_at ? Date.parse(s.fetched_at) : NaN;
  if (Number.isNaN(t) || now - t > STALE_AFTER_MS) return { state: 'stale', poll: true };
  return { state: 'live', poll: true };
}
