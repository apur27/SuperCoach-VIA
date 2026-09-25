import type { LadderRow } from '../../lib/contracts';
import { formatStat } from '../../lib/format';

export function LadderTable({ rows, caption, highlight, firstColumn = 'Pos' }: { rows: LadderRow[]; caption: string; highlight?: string; firstColumn?: string }) {
  if (!rows.length) return <p className="muted">No ladder rows recorded.</p>;
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${caption}`}>
      <table>
        <caption>{caption}</caption>
        <thead><tr><th scope="col" className="num">{firstColumn}</th><th scope="col">Club</th><th scope="col" className="num">P</th><th scope="col" className="num">W</th><th scope="col" className="num">L</th><th scope="col" className="num">D</th><th scope="col" className="num col-optional">For</th><th scope="col" className="num col-optional">Against</th><th scope="col" className="num">%</th><th scope="col" className="num">Pts</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.club_id}-${i}`} aria-current={r.club_id === highlight ? 'true' : undefined}>
              <td className="num">{r.position}</td>
              <th scope="row">{r.name}{r.club_id === highlight ? <strong> (this club)</strong> : null}</th>
              <td className="num">{r.played}</td><td className="num">{r.won}</td><td className="num">{r.lost}</td><td className="num">{r.drawn}</td>
              <td className="num col-optional">{r.points_for}</td><td className="num col-optional">{r.points_against}</td>
              <td className="num">{r.percentage === null ? <span className="missing">n/a</span> : formatStat(r.percentage, 1)}</td>
              <td className="num">{r.premiership_points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
