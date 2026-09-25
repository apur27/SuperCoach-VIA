import type { MatchSummary } from '../../lib/contracts';
import { encodeId, withBase } from '../../lib/ids';
import { formatDateOnly } from '../../lib/format';
import { isReplay, resultLine, statusLabel, teamScoreText } from '../../lib/matches';
import { siteBase } from './runtime';

export function MatchRowsTable({ matches, caption }: { matches: MatchSummary[]; caption: string }) {
  const base = siteBase();
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={`Scrollable table: ${caption}`}>
      <table>
        <caption>{caption}</caption>
        <thead><tr><th scope="col">Match</th><th scope="col">Date</th><th scope="col">Stage</th><th scope="col">Status</th><th scope="col" className="num">Home</th><th scope="col" className="num">Away</th><th scope="col">Result</th></tr></thead>
        <tbody>
          {matches.map((m) => (
            <tr key={m.match_id}>
              <th scope="row"><a href={withBase(base, `match/?id=${encodeId(m.match_id)}`)}>{m.home.name} v {m.away.name}</a></th>
              <td>{m.local_start ? <>{m.local_start} <span className="muted">(local)</span></> : formatDateOnly(m.match_date)}</td>
              <td>{m.stage_label}{isReplay(m) ? <span className="badge badge-muted">Replay</span> : null}</td>
              <td>{statusLabel(m.status)}</td>
              <td className="num">{m.status === 'scheduled' ? '—' : teamScoreText(m.home)}</td>
              <td className="num">{m.status === 'scheduled' ? '—' : teamScoreText(m.away)}</td>
              <td>{resultLine(m)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
