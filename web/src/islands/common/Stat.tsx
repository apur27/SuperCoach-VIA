import { formatStat } from '../../lib/format';

/** A statistic cell: null is "not recorded", never zero. */
export function Stat({ value, digits = 0 }: { value: number | null | undefined; digits?: number }) {
  if (value === null || value === undefined) return <span className="missing">not recorded</span>;
  return <>{formatStat(value, digits)}</>;
}
