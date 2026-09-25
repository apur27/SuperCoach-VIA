/** Intl-based formatting. No manual timezone offsets; date-only values stay date-only. */

export type ZoneChoice = 'Australia/Melbourne' | 'local' | 'UTC';
export const DEFAULT_ZONE: ZoneChoice = 'Australia/Melbourne';
const LOCALE = 'en-AU';

export function resolveZone(zone: ZoneChoice): string | undefined {
  return zone === 'local' ? undefined : zone;
}

export function zoneLabel(zone: ZoneChoice): string {
  if (zone === 'Australia/Melbourne') return 'Melbourne time';
  if (zone === 'UTC') return 'UTC';
  return 'your local time';
}

export function formatInstant(iso: string | null | undefined, zone: ZoneChoice = DEFAULT_ZONE): string {
  if (!iso) return 'not recorded';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'invalid time';
  const tz = resolveZone(zone);
  return new Intl.DateTimeFormat(LOCALE, {
    day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit',
    ...(tz ? { timeZone: tz } : {}),
    timeZoneName: 'short',
  }).format(d);
}

/** A calendar date ("YYYY-MM-DD"): formatted without any timezone conversion. */
export function formatDateOnly(date: string | null | undefined): string {
  if (!date) return 'date unknown';
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  if (!m) return 'date unknown';
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return new Intl.DateTimeFormat(LOCALE, { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(d);
}

export function formatNumber(n: number, digits = 0): string {
  return new Intl.NumberFormat(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n);
}

export const NOT_RECORDED = 'not recorded';

/** Missing statistics are "not recorded", never zero. */
export function formatStat(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return NOT_RECORDED;
  return formatNumber(n, Number.isInteger(n) && digits === 0 ? 0 : digits);
}

export function formatPercent(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return NOT_RECORDED;
  return `${formatNumber(n * 100, digits)}%`;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
}

export function formatScore(goals: number | null, behinds: number | null, score: number | null): string {
  if (score === null) return 'score not recorded';
  if (goals === null || behinds === null) return String(score);
  return `${goals}.${behinds} (${score})`;
}
