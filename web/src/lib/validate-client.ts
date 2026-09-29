/** Browser-side validation: loads only the validator for the requested resource kind. */
import type { ResourceKind, ResourceTypes } from './contracts.generated';
import { VALIDATOR_LOADERS } from './validators/index.generated';
import { compactRowProblem } from './stats';

/** Cross-field rules a JSON Schema cannot state (row width against stat_columns). */
function semanticProblem(kind: ResourceKind, value: unknown): string | null {
  if (kind === 'match_detail') {
    const d = value as ResourceTypes['match_detail'];
    return compactRowProblem(d.stat_columns, d.home_players.stats) ?? compactRowProblem(d.stat_columns, d.away_players.stats);
  }
  if (kind === 'player_season_games') {
    const g = value as ResourceTypes['player_season_games'];
    return compactRowProblem(g.stat_columns, g.games.stats);
  }
  return null;
}

export async function validateAsync<K extends ResourceKind>(kind: K, data: unknown): Promise<{ ok: true; value: ResourceTypes[K] } | { ok: false; error: string }> {
  const fn = await VALIDATOR_LOADERS[kind]();
  if (fn(data)) {
    const problem = semanticProblem(kind, data);
    return problem ? { ok: false, error: `/stat_columns ${problem}` } : { ok: true, value: data };
  }
  const e = fn.errors?.[0];
  return { ok: false, error: e ? `${e.instancePath || '/'} ${e.message ?? e.keyword}` : 'invalid' };
}
