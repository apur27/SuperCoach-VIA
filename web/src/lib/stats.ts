/**
 * Expanders for the compact player resources. Mean and coverage are not shipped: they are
 * exactly total / observed_games and min(1, observed_games / scope games), the same
 * arithmetic as the Python view models (publish/view_models.expand_stats).
 */
import type { BoxScoreColumns, PlayerGameColumns, StatColumns, StatValue } from './contracts';

export function expandStats(names: readonly string[], cols: StatColumns, scopeGames: number): StatValue[] {
  if (names.length !== cols.total.length) throw new Error('stat_names and StatColumns differ in length');
  return names.map((stat, i) => {
    const total = cols.total[i] ?? null;
    const observed = cols.observed_games[i] ?? 0;
    return {
      stat,
      total,
      mean: total === null || observed === 0 ? null : total / observed,
      observed_games: observed,
      eligible_games: cols.eligible_games[i] ?? 0,
      coverage: scopeGames <= 0 ? null : Math.min(1, observed / scopeGames),
    };
  });
}

export interface GameRow {
  match_id: string;
  match_date: string | null;
  date_quality: PlayerGameColumns['date_quality'][number];
  stage_label: string;
  club_id: string;
  opponent_club_id: string | null;
  opponent_name: string | null;
  result: string | null;
  career_game_counter: number | null;
  stats: (number | null)[];
}

export function gameRows(g: PlayerGameColumns): GameRow[] {
  return g.match_id.map((match_id, i) => ({
    match_id,
    match_date: g.match_date[i] ?? null,
    date_quality: g.date_quality[i] ?? 'unknown',
    stage_label: g.stage_label[i] ?? '',
    club_id: g.club_id[i] ?? '',
    opponent_club_id: g.opponent_club_id[i] ?? null,
    opponent_name: g.opponent_name[i] ?? null,
    result: g.result[i] ?? null,
    career_game_counter: g.career_game_counter[i] ?? null,
    stats: g.stats[i] ?? [],
  }));
}

export interface BoxRow { player_id: string; name: string; stats: (number | null)[] }

export function boxRows(b: BoxScoreColumns): BoxRow[] {
  return b.player_id.map((player_id, i) => ({ player_id, name: b.name[i] ?? player_id, stats: b.stats[i] ?? [] }));
}
