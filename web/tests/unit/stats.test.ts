import { describe, expect, it } from 'vitest';
import { applyMatchFacts, boxRows, compactRowProblem, expandStats, gameRows } from '../../src/lib/stats';

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
  it('fills date, stage and opponent from the match index when the log omitted them', () => {
    const rows = gameRows({
      match_id: ['m1'], match_date: [], date_quality: ['source'], stage_label: [], club_id: ['a'],
      opponent_club_id: [], opponent_name: [], result: ['W'], career_game_counter: [1], stats: [[10]],
    });
    const filled = applyMatchFacts(rows, {
      matches: [{
        match_id: 'm1', match_date: '2026-03-07', stage_label: 'Round 1',
        home: { club_id: 'a', name: 'Demo Harbour' }, away: { club_id: 'b', name: 'Demo Ridge' },
      }],
    });
    expect(filled[0]).toMatchObject({ match_date: '2026-03-07', stage_label: 'Round 1', opponent_name: 'Demo Ridge', result: 'W' });
  });
  it('restores box-score rows from columns', () => {
    expect(boxRows({ player_id: ['p1', 'p2'], name: ['A', 'B'], stats: [[1, null], [0, 3]] })).toEqual([
      { player_id: 'p1', name: 'A', stats: [1, null] }, { player_id: 'p2', name: 'B', stats: [0, 3] }]);
  });
});

describe('compact row contract (O55-04)', () => {
  it('accepts rows as wide as their stat_columns', () => {
    expect(compactRowProblem(['kicks', 'goals'], [[1, 2], [null, 0]])).toBeNull();
  });
  it('refuses a short or long row, which would shift values onto the wrong statistic', () => {
    expect(compactRowProblem(['kicks', 'goals'], [[1]])).toMatch(/row 0 has 1 values for 2/);
    expect(compactRowProblem(['kicks', 'goals'], [[1, 2, 3]])).toMatch(/row 0 has 3/);
  });
  it('refuses duplicate stat columns', () => {
    expect(compactRowProblem(['kicks', 'kicks'], [[1, 2]])).toMatch(/duplicate/);
  });
});
