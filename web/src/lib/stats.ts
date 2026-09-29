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

/** Fill date, stage and opponent from the season match index when the log left those arrays empty. */
export function applyMatchFacts(rows: GameRow[], index: { matches: { match_id: string; match_date: string | null; stage_label: string; home: { club_id: string; name: string }; away: { club_id: string; name: string } }[] } | null): GameRow[] {
  if (!index) return rows;
  const byId = new Map(index.matches.map((m) => [m.match_id, m]));
  return rows.map((r) => {
    if (r.stage_label) return r;
    const m = byId.get(r.match_id);
    if (!m) return r;
    const opp = r.club_id === m.home.club_id ? m.away : r.club_id === m.away.club_id ? m.home : null;
    return {
      ...r,
      match_date: r.match_date ?? m.match_date,
      stage_label: m.stage_label,
      opponent_club_id: r.opponent_club_id ?? opp?.club_id ?? null,
      opponent_name: r.opponent_name ?? opp?.name ?? null,
    };
  });
}

export interface BoxRow { player_id: string; name: string; stats: (number | null)[] }

export function boxRows(b: BoxScoreColumns): BoxRow[] {
  return b.player_id.map((player_id, i) => ({ player_id, name: b.name[i] ?? player_id, stats: b.stats[i] ?? [] }));
}

/**
 * Compact-row contract: each stats row has exactly one value per declared stat column and no
 * column repeats. A short row would move every later value onto the wrong statistic.
 * (Vocabulary and canonical order are enforced by the Python view models and the checker.)
 */
export function compactRowProblem(columns: readonly string[], rows: readonly (readonly (number | null)[])[]): string | null {
  if (new Set(columns).size !== columns.length) return 'duplicate stat_columns';
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    if (row.length !== columns.length) return `row ${i} has ${row.length} values for ${columns.length} stat_columns`;
  }
  return null;
}
