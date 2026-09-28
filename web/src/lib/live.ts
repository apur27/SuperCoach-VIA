export const POLL_MS = 90_000;
/** An in-progress snapshot older than this is shown as delayed/stale. */
export const STALE_AFTER_MS = 5 * 60_000;

export function liveFreshness(s: { final: boolean; fetched_at: string | null }, now: number): { state: 'final' | 'snapshot' | 'stale'; poll: boolean } {
  // A published release is an archived snapshot. It is not a live feed, so the page
  // never polls it. POLL_MS remains the local collector's interval, not a site cadence.
  if (s.final) return { state: 'final', poll: false };
  const t = s.fetched_at ? Date.parse(s.fetched_at) : NaN;
  if (Number.isNaN(t) || now - t > STALE_AFTER_MS) return { state: 'stale', poll: false };
  return { state: 'snapshot', poll: false };
}
