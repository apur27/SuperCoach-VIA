/** Public data contract: generated types + generated runtime validators. */
export type * from './contracts.generated';
import type { ResourceKind, ResourceTypes } from './contracts.generated';
import * as V from './validators.generated.js';
import type { GeneratedValidator } from './validators.generated.js';

const VALIDATORS: { [K in ResourceKind]: GeneratedValidator<ResourceTypes[K]> } = {
  accuracy_index: V.validateAccuracyIndex,
  accuracy_report: V.validateAccuracyReport,
  article: V.validateArticle,
  article_index: V.validateArticleIndex,
  downloads: V.validateDownloads,
  history_index: V.validateHistoryIndex,
  history_table: V.validateHistoryTable,
  lists_index: V.validateListsIndex,
  lists_season: V.validateListsSeason,
  live_index: V.validateLiveIndex,
  live_snapshot: V.validateLiveSnapshot,
  match_detail: V.validateMatchDetail,
  match_index: V.validateMatchIndex,
  overview: V.validateOverview,
  player_detail: V.validatePlayerDetail,
  player_index: V.validatePlayerIndex,
  player_season_games: V.validatePlayerSeasonGames,
  prediction_index: V.validatePredictionIndex,
  prediction_set: V.validatePredictionSet,
  quality: V.validateQuality,
  release: V.validateRelease,
  team_index: V.validateTeamIndex,
  team_season: V.validateTeamSeason,
};

export type ValidationResult<T> = { ok: true; value: T } | { ok: false; error: string };

export function validate<K extends ResourceKind>(kind: K, data: unknown): ValidationResult<ResourceTypes[K]> {
  const fn = VALIDATORS[kind];
  if (fn(data)) return { ok: true, value: data };
  const e = fn.errors?.[0];
  return { ok: false, error: e ? `${e.instancePath || '/'} ${e.message ?? e.keyword}` : 'invalid' };
}

/** Which schema a release-relative path must satisfy (release layout contract). */
export function kindForPath(path: string): ResourceKind | null {
  const rules: [RegExp, ResourceKind][] = [
    [/^release\.json$/, 'release'],
    [/^overview\.json$/, 'overview'],
    [/^quality\.json$/, 'quality'],
    [/^downloads\.json$/, 'downloads'],
    [/^predictions\/index\.json$/, 'prediction_index'],
    [/^predictions\/[^/]+\/[^/]+\.json$/, 'prediction_set'],
    [/^players\/index\.json$/, 'player_index'],
    [/^players\/[^/]+\.json$/, 'player_detail'],
    [/^player-games\/[^/]+\/[^/]+\.json$/, 'player_season_games'],
    [/^teams\/index\.json$/, 'team_index'],
    [/^teams\/[^/]+\/[^/]+\.json$/, 'team_season'],
    [/^matches\/detail\/[^/]+\.json$/, 'match_detail'],
    [/^matches\/[^/]+\/index\.json$/, 'match_index'],
    [/^history\/index\.json$/, 'history_index'],
    [/^history\/[^/]+\/[^/]+\.json$/, 'history_table'],
    [/^accuracy\/index\.json$/, 'accuracy_index'],
    [/^accuracy\/[^/]+\/[^/]+\.json$/, 'accuracy_report'],
    [/^lists\/index\.json$/, 'lists_index'],
    [/^lists\/[^/]+\.json$/, 'lists_season'],
    [/^articles\/index\.json$/, 'article_index'],
    [/^articles\/[^/]+\.json$/, 'article'],
    [/^live\/index\.json$/, 'live_index'],
    [/^live\/[^/]+\/[^/]+\.json$/, 'live_snapshot'],
  ];
  for (const [re, kind] of rules) if (re.test(path)) return kind;
  return null;
}
