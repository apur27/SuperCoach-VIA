import type { StatValue } from '../../lib/contracts';
import { formatPercent } from '../../lib/format';
import { Stat } from './Stat';

/** StatValues with denominators and coverage; null totals are "not recorded". */
export function StatTable({ caption, stats }: { caption: string; stats: StatValue[] }) {
  if (!stats.length) return <p className="muted">No statistics recorded.</p>;
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${caption}`}>
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr><th scope="col">Statistic</th><th scope="col" className="num">Total</th><th scope="col" className="num">Per game</th><th scope="col" className="num">Games with data</th><th scope="col" className="num">Coverage</th></tr>
        </thead>
        <tbody>
          {stats.map((s) => (
            <tr key={s.stat}>
              <th scope="row">{s.stat}</th>
              <td className="num"><Stat value={s.total} /></td>
              <td className="num"><Stat value={s.mean} digits={1} /></td>
              <td className="num">{s.observed_games} of {s.eligible_games}</td>
              <td className="num">{s.coverage === null ? <span className="missing">not recorded</span> : formatPercent(s.coverage)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
