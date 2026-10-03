// Totals, denominators, means and missing labels as the browser renders them, against values
// worked out by hand from the compact columns (not by calling the code under test).
import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { StatTable } from '../../src/islands/common/StatTable';
import { expandStats } from '../../src/lib/stats';

function cells(html: string, stat: string): string[] {
  const row = new RegExp(`<tr><th scope="row"[^>]*>${stat}</th>(.*?)</tr>`).exec(html);
  if (!row) throw new Error(`no row for ${stat}`);
  return [...row[1]!.matchAll(/<td class="num">(.*?)<\/td>/g)].map((m) => m[1]!.replace(/<[^>]+>/g, ''));
}

describe('StatTable', () => {
  // 20 career games. goals: 9 over all 20 (blanks resolved to recorded zeros) -> 0.45 -> "0.5".
  // tackles: recorded in 10 of the 12 games inside its era -> 34 / 10 = 3.4, coverage 10/20.
  // hitouts: never recorded -> total and mean "not recorded", 0 of 0.
  // bounces: recorded zero in every game -> 0 and 0.0, not "not recorded".
  const html = renderToStaticMarkup(
    <StatTable
      caption="Career"
      stats={expandStats(
        ['goals', 'tackles', 'hitouts', 'bounces'],
        { total: [9, 34, null, 0], observed_games: [20, 10, 0, 20], eligible_games: [20, 12, 0, 20] },
        20,
      )}
    />,
  );

  it('divides a total by the games with data and shows that denominator', () => {
    expect(cells(html, 'Goals')).toEqual(['9', '0.5', '20 of 20', '100.0%']);
  });
  it('keeps a genuine coverage gap visible as N of M', () => {
    expect(cells(html, 'Tackles')).toEqual(['34', '3.4', '10 of 12', '50.0%']);
  });
  it('labels an unrecorded statistic "not recorded", never zero', () => {
    expect(cells(html, 'Hitouts')).toEqual(['not recorded', 'not recorded', '0 of 0', '0.0%']);
  });
  it('shows a recorded zero as zero', () => {
    expect(cells(html, 'Bounces')).toEqual(['0', '0.0', '20 of 20', '100.0%']);
  });
  it('uses readable statistic names while keeping values and denominators', () => {
    const labels = renderToStaticMarkup(<StatTable caption="Career" stats={expandStats(['goal_assists'], { total: [9], observed_games: [10], eligible_games: [10] }, 10)} />);
    expect(cells(labels, 'Goal assists')).toEqual(['9', '0.9', '10 of 10', '100.0%']);
  });
  it('names the denominator column', () => {
    expect(html).toContain('<th scope="col" class="num">Games with data</th>');
  });
});
