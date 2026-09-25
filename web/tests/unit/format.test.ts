import { describe, expect, it } from 'vitest';
import { formatInstant, formatDateOnly, formatStat, formatBytes, formatNumber, formatPercent, zoneLabel } from '../../src/lib/format';

describe('format', () => {
  it('formats instants in Melbourne with an explicit zone label', () => {
    const s = formatInstant('2026-09-24T22:00:00Z', 'Australia/Melbourne');
    expect(s).toContain('25 Sept 2026');
    expect(s).toMatch(/8:00\s?am/i);
    expect(s).toMatch(/AEST|GMT\+10|Melbourne/);
  });
  it('formats instants in UTC when asked', () => {
    expect(formatInstant('2026-09-24T22:00:00Z', 'UTC')).toMatch(/24 Sept 2026.*10:00\s?pm.*UTC/i);
  });
  it('keeps date-only values date-only regardless of zone', () => {
    expect(formatDateOnly('2026-03-01')).toBe('1 Mar 2026');
    expect(formatDateOnly(null)).toBe('date unknown');
  });
  it('never renders a missing statistic as zero', () => {
    expect(formatStat(null)).toBe('not recorded');
    expect(formatStat(undefined)).toBe('not recorded');
    expect(formatStat(0)).toBe('0');
    expect(formatStat(12.345, 1)).toBe('12.3');
  });
  it('formats numbers, percents and bytes', () => {
    expect(formatNumber(1234.5, 1)).toBe('1,234.5');
    expect(formatPercent(0.805)).toBe('80.5%');
    expect(formatPercent(null)).toBe('not recorded');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.0 KiB');
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MiB');
  });
  it('labels zones', () => {
    expect(zoneLabel('Australia/Melbourne')).toBe('Melbourne time');
    expect(zoneLabel('UTC')).toBe('UTC');
  });
});
