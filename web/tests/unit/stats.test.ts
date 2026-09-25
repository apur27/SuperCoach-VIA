import { describe, expect, it } from 'vitest';
import { expandStats, gameRows } from '../../src/lib/stats';

describe('compact player resources', () => {
  it('derives mean and coverage exactly; unknown stays null and zero stays zero', () => {
    const v = expandStats(['disposals', 'goals', 'tackles', 'brownlow_votes'],
      { total: [412, 7, null, 0], observed_games: [18, 3, 0, 18], eligible_games: [18, 18, 0, 18] }, 20);
    expect(v[0]).toEqual({ stat: 'disposals', total: 412, mean: 412 / 18, observed_games: 18, eligible_games: 18, coverage: 18 / 20 });
    expect(v[1]!.mean).toBe(7 / 3);
    expect(v[2]).toMatchObject({ total: null, mean: null, coverage: 0 });
    expect(v[3]).toMatchObject({ total: 0, mean: 0 });
    expect(expandStats(['x'], { total: [1], observed_games: [1], eligible_games: [1] }, 0)[0]!.coverage).toBeNull();
  });
  it('refuses misaligned stat names', () => {
    expect(() => expandStats(['a', 'b'], { total: [1], observed_games: [1], eligible_games: [1] }, 1)).toThrow(/length/);
  });
  it('restores game rows from columns', () => {
    const rows = gameRows({ match_id: ['m1', 'm2'], match_date: ['2026-03-07', null], date_quality: ['fixture_verified', 'unknown'],
      stage_label: ['1', 'QF'], club_id: ['a', 'a'], opponent_club_id: ['b', null], opponent_name: ['B', null], result: ['W', null],
      career_game_counter: [1, null], stats: [[10, null], [0, 2]] });
    expect(rows).toHaveLength(2);
    expect(rows[1]).toEqual({ match_id: 'm2', match_date: null, date_quality: 'unknown', stage_label: 'QF', club_id: 'a',
      opponent_club_id: null, opponent_name: null, result: null, career_game_counter: null, stats: [0, 2] });
  });
});
