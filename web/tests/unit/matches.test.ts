import { describe, expect, it } from 'vitest';
import { statusLabel, teamScoreText, resultLine, reasonText, isReplay } from '../../src/lib/matches';
import type { MatchSummary } from '../../src/lib/contracts';

const team = (club_id: string, goals: number | null, behinds: number | null) => ({ club_id, name: club_id, goals, behinds, score: goals === null || behinds === null ? null : goals * 6 + behinds });
const m = (o: Partial<MatchSummary>): MatchSummary => ({
  match_id: 'x', season: 2026, stage_id: 'r01', stage_label: 'Round 1', stage_type: 'regular', round_number: 1, stage_order: 1, replay_occurrence: 1,
  local_start: null, match_date: null, date_precision: 'unknown', status: 'complete', venue: null, home: team('a', 1, 1), away: team('b', 0, 0), winner_club_id: 'a', ...o,
});

describe('match display', () => {
  it('distinguishes a zero score from a missing score', () => {
    expect(teamScoreText(team('a', 0, 0))).toBe('0.0 (0)');
    expect(teamScoreText(team('a', null, null))).toBe('score not recorded');
  });
  it('labels statuses honestly', () => {
    expect(statusLabel('postponed')).toBe('Postponed');
    expect(statusLabel('unknown')).toBe('Status unknown');
    expect(statusLabel('scheduled')).toBe('Scheduled');
  });
  it('describes results including draws and missing scores', () => {
    expect(resultLine(m({}))).toBe('a won by 7');
    expect(resultLine(m({ home: team('a', 1, 1), away: team('b', 1, 1), winner_club_id: null }))).toBe('Draw');
    expect(resultLine(m({ status: 'unknown', home: team('a', null, null), away: team('b', null, null), winner_club_id: null }))).toBe('Result not recorded');
    expect(resultLine(m({ status: 'scheduled', winner_club_id: null }))).toBe('Not yet played');
    expect(resultLine(m({ status: 'postponed', winner_club_id: null }))).toBe('Postponed');
  });
  it('flags replays', () => {
    expect(isReplay(m({ replay_occurrence: 2 }))).toBe(true);
    expect(isReplay(m({}))).toBe(false);
  });
  it('explains forecast unavailability', () => {
    expect(reasonText('no_valid_future_fixture')).toMatch(/No valid future fixture/);
    expect(reasonText(null)).toBe('No reason was recorded.');
    expect(reasonText('custom reason text')).toBe('custom reason text');
  });
});
