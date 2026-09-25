// Writes the DEMO fixture release used when SCVIA_RELEASE_DIR is unset.
// Every value is synthetic and labelled DEMO: club and player names say "Demo", stat
// values follow obvious arithmetic patterns, and dataset_status is "demo". It exists to
// exercise the browser against every resource type and edge case (same-name players,
// diacritics, missing vs zero scores, postponement, drawn final + replay, null coverage,
// unavailable/expired forecasts), never to resemble real AFL statistics.
//
// Output is deterministic (no clock reads). Run: node scripts/make-demo-fixture.mjs
import { createHash } from 'node:crypto';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { crc32 } from 'node:zlib';
import * as V from '../src/lib/validators.generated.js';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '../tests/fixtures/demo-release');
const RELEASE_ID = '20260925T000000Z-demo';
const SNAPSHOT_ID = 'demo-snapshot-0001';
const GEN = '2026-09-25T00:00:00Z';
const SEASON = 2026;
const DEMO_NOTE = 'DEMO data: synthetic values for interface testing only; not AFL statistics.';

const written = new Map(); // rel path -> bytes
function canonical(obj) {
  const sort = (v) =>
    Array.isArray(v)
      ? v.map(sort)
      : v && typeof v === 'object'
        ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sort(v[k])]))
        : v;
  return Buffer.from(JSON.stringify(sort(obj)) + '\n', 'utf8');
}
const validators = {
  release: V.validateRelease, overview: V.validateOverview, prediction_index: V.validatePredictionIndex,
  prediction_set: V.validatePredictionSet, player_index: V.validatePlayerIndex, player_detail: V.validatePlayerDetail,
  player_season_games: V.validatePlayerSeasonGames, team_index: V.validateTeamIndex, team_season: V.validateTeamSeason,
  match_index: V.validateMatchIndex, match_detail: V.validateMatchDetail, history_index: V.validateHistoryIndex,
  history_table: V.validateHistoryTable, accuracy_index: V.validateAccuracyIndex, accuracy_report: V.validateAccuracyReport,
  lists_index: V.validateListsIndex, lists_season: V.validateListsSeason, article_index: V.validateArticleIndex,
  article: V.validateArticle, live_index: V.validateLiveIndex, live_snapshot: V.validateLiveSnapshot,
  quality: V.validateQuality, downloads: V.validateDownloads,
};
function put(rel, kind, obj) {
  const v = validators[kind];
  if (!v(obj)) throw new Error(`${rel} invalid as ${kind}: ${JSON.stringify(v.errors)}`);
  const bytes = canonical(obj);
  written.set(rel, bytes);
  return rel;
}
function putRaw(rel, bytes) {
  written.set(rel, Buffer.from(bytes));
  return rel;
}
const sha = (b) => createHash('sha256').update(b).digest('hex');
const ref = (rel) => ({ path: rel, sha256: sha(written.get(rel)), bytes: written.get(rel).length });

// ---------------------------------------------------------------- clubs
const CLUBS = [
  { club_id: 'demo_a', name: 'Demo Club A', first: 2000, last: 2026, active: true },
  { club_id: 'demo_b', name: 'Demo Club B', first: 2000, last: 2026, active: true },
  { club_id: 'demo_c', name: 'Demo Club C', first: 2000, last: 2026, active: true },
  { club_id: 'demo_d', name: 'Demo Club D', first: 2000, last: 2026, active: true },
  { club_id: 'demo_old', name: 'Demo Club Old (historical)', first: 1990, last: 1995, active: false },
];
const club = (id) => CLUBS.find((c) => c.club_id === id);
const clubRef = (id) => ({ club_id: id, name: club(id).name });

// ---------------------------------------------------------------- matches
function team(id, goals, behinds) {
  const score = goals === null || behinds === null ? null : goals * 6 + behinds;
  return { club_id: id, name: club(id).name, goals, behinds, score };
}
function match(o) {
  const home = team(o.home, o.hg ?? null, o.hb ?? null);
  const away = team(o.away, o.ag ?? null, o.ab ?? null);
  let winner = null;
  if (o.status === 'complete' && home.score !== null && away.score !== null && home.score !== away.score) {
    winner = home.score > away.score ? home.club_id : away.club_id;
  }
  return {
    match_id: o.id, season: o.season ?? SEASON, stage_id: o.stage, stage_label: o.label,
    stage_type: o.type ?? 'regular', round_number: o.round ?? null, stage_order: o.order,
    replay_occurrence: o.replay ?? 1, local_start: o.local ?? null, match_date: o.date ?? null,
    date_precision: o.local ? 'minute' : o.date ? 'day' : 'unknown', status: o.status,
    venue: o.venue ?? 'Demo Ground', home, away, winner_club_id: winner,
  };
}
const M2026 = [
  match({ id: 'demo:2026:r01:a-b', stage: 'r01', label: 'Demo Round 1', round: 1, order: 1, local: '2026-03-14 19:30', date: '2026-03-14', status: 'complete', home: 'demo_a', away: 'demo_b', hg: 10, hb: 10, ag: 5, ab: 5 }),
  match({ id: 'demo:2026:r01:c-d', stage: 'r01', label: 'Demo Round 1', round: 1, order: 1, local: '2026-03-14 19:30', date: '2026-03-14', status: 'complete', home: 'demo_c', away: 'demo_d', hg: 0, hb: 0, ag: 1, ab: 1 }),
  match({ id: 'demo:2026:r02:a-c', stage: 'r02', label: 'Demo Round 2', round: 2, order: 2, local: '2026-03-21 13:10', date: '2026-03-21', status: 'complete', home: 'demo_a', away: 'demo_c', hg: 8, hb: 8, ag: 8, ab: 8 }),
  match({ id: 'demo:2026:r02:b-d', stage: 'r02', label: 'Demo Round 2', round: 2, order: 2, date: '2026-03-21', status: 'unknown', home: 'demo_b', away: 'demo_d' }),
  match({ id: 'demo:2026:r03:a-d', stage: 'r03', label: 'Demo Round 3', round: 3, order: 3, local: '2026-03-28 16:15', date: '2026-03-28', status: 'postponed', home: 'demo_a', away: 'demo_d' }),
  match({ id: 'demo:2026:r03:b-c', stage: 'r03', label: 'Demo Round 3', round: 3, order: 3, local: '2026-03-28 16:15', date: '2026-03-28', status: 'complete', home: 'demo_b', away: 'demo_c', hg: 6, hb: 6, ag: 7, ab: 7 }),
  match({ id: 'demo:2026:qf1:a-b', stage: 'qf1', label: 'Demo Qualifying Final', type: 'final', order: 10, local: '2026-09-05 19:40', date: '2026-09-05', status: 'complete', home: 'demo_a', away: 'demo_b', hg: 9, hb: 9, ag: 9, ab: 9 }),
  match({ id: 'demo:2026:qf1:a-b:replay', stage: 'qf1', label: 'Demo Qualifying Final (replay)', type: 'final', order: 10, replay: 2, local: '2026-09-12 19:40', date: '2026-09-12', status: 'complete', home: 'demo_a', away: 'demo_b', hg: 11, hb: 11, ag: 10, ab: 10 }),
  match({ id: 'demo:2026:r05:a-c', stage: 'r05', label: 'Demo Round 5', round: 5, order: 5, local: '2026-10-03 19:30', date: '2026-10-03', status: 'scheduled', home: 'demo_a', away: 'demo_c' }),
  match({ id: 'demo:2026:r05:b-d', stage: 'r05', label: 'Demo Round 5', round: 5, order: 5, local: '2026-10-04 15:20', date: '2026-10-04', status: 'scheduled', home: 'demo_b', away: 'demo_d' }),
];
const M2025 = [
  match({ id: 'demo:2025:r01:a-b', season: 2025, stage: 'r01', label: 'Demo Round 1', round: 1, order: 1, date: '2025-03-15', status: 'complete', home: 'demo_a', away: 'demo_b', hg: 12, hb: 12, ag: 6, ab: 6 }),
  match({ id: 'demo:2025:r01:c-d', season: 2025, stage: 'r01', label: 'Demo Round 1', round: 1, order: 1, date: '2025-03-15', status: 'complete', home: 'demo_c', away: 'demo_d', hg: 7, hb: 7, ag: 7, ab: 8 }),
];
const M1994 = [
  match({ id: 'demo:1994:r01:old-a', season: 1994, stage: 'r01', label: 'Demo Round 1', round: 1, order: 1, date: '1994-04-02', status: 'complete', home: 'demo_old', away: 'demo_a', hg: 15, hb: 15, ag: 10, ab: 10 }),
];
const ALL_MATCHES = [...M2026, ...M2025, ...M1994];
const matchKey = (id) => id.replaceAll(':', '__');
const upcoming = M2026.filter((m) => m.status === 'scheduled');
const recent = M2026.filter((m) => m.status === 'complete').slice(-3).reverse();

// ---------------------------------------------------------------- players
const STAT_COLS = ['disposals', 'kicks', 'handballs', 'goals', 'tackles'];
function statLine(base, games, coverage = 1) {
  const observed = Math.round(games * coverage);
  return STAT_COLS.map((stat, i) => {
    const per = base + i;
    if (coverage === null || observed === 0) {
      return { stat, total: null, mean: null, observed_games: 0, eligible_games: games, coverage: coverage === null ? null : 0 };
    }
    return { stat, total: per * observed, mean: per, observed_games: observed, eligible_games: games, coverage };
  });
}
/** Compact player-page stats (StatColumns): mean and coverage are derived by the client. */
function statColumns(base, games, coverage = 1) {
  const lines = statLine(base, games, coverage);
  return { total: lines.map((l) => l.total), observed_games: lines.map((l) => l.observed_games), eligible_games: lines.map((l) => l.eligible_games) };
}
const GAME_FIELDS = ['match_id', 'match_date', 'date_quality', 'stage_label', 'club_id', 'opponent_club_id', 'opponent_name', 'result', 'career_game_counter', 'stats'];
const gameColumns = (rows) => Object.fromEntries(GAME_FIELDS.map((f) => [f, rows.map((r) => r[f])]));
const PLAYERS = [];
function player(o) {
  const id = `legacy:${o.slug}`;
  const p = { id, key: id.replaceAll(':', '__'), ...o };
  PLAYERS.push(p);
  return p;
}
player({ slug: 'demo_player_a1', name: 'Demo Player A1', first: 'Demo', last: 'Player A1', clubs: ['demo_a'], seasons: [2025, 2026], base: 20, active: true });
player({ slug: 'demo_player_b1', name: 'Demo Player B1', first: 'Demo', last: 'Player B1', clubs: ['demo_b'], seasons: [2026], base: 15, active: true });
player({ slug: 'demo_same_name_2020', name: 'Demo Same Name', first: 'Demo', last: 'Same Name', clubs: ['demo_a', 'demo_c'], seasons: [2025, 2026], base: 12, active: true });
player({ slug: 'demo_same_name_1990', name: 'Demo Same Name', first: 'Demo', last: 'Same Name', clubs: ['demo_old'], seasons: [1994], base: 10, active: false, coverage: 0.5 });
player({ slug: 'demo_zoe_arger', name: 'Demo Zoë Ärger', first: 'Demo Zoë', last: 'Ärger', clubs: ['demo_c'], seasons: [2026], base: 18, active: true });
player({ slug: 'demo_player_sparse', name: 'Demo Player Sparse', first: 'Demo', last: 'Player Sparse', clubs: ['demo_old'], seasons: [1994], base: 5, active: false, coverage: null });
for (let i = 1; i <= 60; i += 1) {
  const n = String(i).padStart(2, '0');
  const c = CLUBS[i % 4].club_id;
  player({ slug: `demo_filler_${n}`, name: `Demo Filler ${n}`, first: 'Demo', last: `Filler ${n}`, clubs: [c], seasons: [2026], base: 10 + (i % 10), active: true, filler: true });
}
const GAMES_PER_SEASON = 4;
function stripDiacritics(s) {
  return s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase();
}

// ---------------------------------------------------------------- predictions (built before players: detail embeds forecast)
function predRow(p, m, i, withInterval) {
  const opp = m.home.club_id === p.clubs[0] ? m.away : m.home;
  const predicted = p.base + 0.5;
  return {
    prediction_id: `demo-pred-${p.slug}-${m.stage_id}`, prediction_run_id: 'demo-run-0001', snapshot_id: SNAPSHOT_ID,
    model_id: 'demo_model_v1', player_id: p.id, player_name: p.name, club_id: p.clubs[0], club_name: club(p.clubs[0]).name,
    opponent_club_id: opp.club_id, opponent_name: opp.name, match_id: m.match_id, season: SEASON, stage_id: m.stage_id,
    stage_label: m.stage_label,
    scheduled_at: null,
    scheduled_local: m.local_start, venue: m.venue, forecast_cutoff: '2026-09-24T00:00:00Z', origin: 'prospective',
    generated_at: GEN, selection_status: i % 3 === 0 ? 'unconfirmed' : 'confirmed', eligibility_basis: 'DEMO: listed in demo fixture',
    history_games: i % 5 === 0 ? 0 : 8, recent_mean_5: i % 5 === 0 ? null : p.base,
    predicted_disposals: predicted,
    interval_low: withInterval ? predicted - 5 : null, interval_high: withInterval ? predicted + 5 : null,
    interval_level: withInterval ? 0.8 : null, interval_method: withInterval ? 'DEMO split-conformal' : null,
    warnings: i % 5 === 0 ? ['DEMO: no prior games; labelled baseline used'] : [],
  };
}
// Scheduled r05 dates are local Melbourne wall times; the UTC instant is fixed explicitly below.
const R05_UTC = { 'demo:2026:r05:a-c': '2026-10-03T09:30:00Z', 'demo:2026:r05:b-d': '2026-10-04T05:20:00Z' };
const MODEL = {
  model_id: 'demo_model_v1', kind: 'model', name: 'DEMO model v1', description: 'DEMO: synthetic model entry for interface testing.',
  trained_cutoff: '2026-09-20T00:00:00Z', promoted: true, promotion_note: 'DEMO: promoted over the demo baseline on demo data.',
};
const activeFixturePlayers = PLAYERS.filter((p) => p.active && p.seasons.includes(2026));
const r05Rows = activeFixturePlayers.slice(0, 14).map((p, i) => {
  const m = upcoming.find((mm) => mm.home.club_id === p.clubs[0] || mm.away.club_id === p.clubs[0]);
  const row = predRow(p, m, i, true);
  row.scheduled_at = R05_UTC[m.match_id];
  return row;
});
const r04Rows = activeFixturePlayers.slice(0, 4).map((p, i) => ({
  ...predRow(p, M2026[6], i, false), stage_id: 'qf1', stage_label: 'Demo Qualifying Final', scheduled_at: '2026-09-05T09:40:00Z',
  scheduled_local: '2026-09-05 19:40', match_id: M2026[6].match_id, prediction_id: `demo-pred-${p.slug}-qf1`,
  opponent_club_id: p.clubs[0] === 'demo_b' ? 'demo_a' : 'demo_b', opponent_name: p.clubs[0] === 'demo_b' ? 'Demo Club A' : 'Demo Club B',
}));
const forecastByPlayer = new Map(r05Rows.map((r) => [r.player_id, r]));

// ---------------------------------------------------------------- player resources
const indexEntries = [];
for (const p of PLAYERS) {
  const seasonLines = [];
  let careerGames = 0;
  for (const season of p.seasons) {
    const games = [];
    for (let g = 1; g <= GAMES_PER_SEASON; g += 1) {
      const pool = ALL_MATCHES.filter((m) => m.season === season);
      const m = pool[(g - 1) % pool.length];
      const stats = STAT_COLS.map((_, i) => (p.coverage === null ? null : p.coverage === 0.5 && g % 2 === 0 ? null : p.base + i + (g % 3)));
      games.push({
        match_id: m.match_id, match_date: m.match_date, date_quality: season < 2000 ? 'inferred' : 'fixture_verified',
        stage_label: m.stage_label, club_id: p.clubs[0],
        opponent_club_id: m.away.club_id === p.clubs[0] ? m.home.club_id : m.away.club_id,
        opponent_name: m.away.club_id === p.clubs[0] ? m.home.name : m.away.name,
        result: m.status === 'complete' ? (g % 2 ? 'W' : 'L') : null, career_game_counter: careerGames + g, stats,
      });
    }
    const gamesRel = `player-games/${p.key}/${season}.json`;
    put(gamesRel, 'player_season_games', { player_id: p.id, season, stat_columns: STAT_COLS, games: gameColumns(games) });
    seasonLines.push({ season, clubs: p.clubs, games: GAMES_PER_SEASON, stats: statColumns(p.base, GAMES_PER_SEASON, p.coverage === undefined ? 1 : p.coverage), games_resource: gamesRel });
    careerGames += GAMES_PER_SEASON;
  }
  const detail = {
    id: p.id, key: p.key, name: p.name, first_name: p.first, last_name: p.last,
    birth_date: p.filler ? null : p.seasons[0] < 2000 ? '1970-01-01' : '2000-01-01',
    birth_date_quality: p.filler ? 'unknown' : p.seasons[0] < 2000 ? 'legacy_filename' : 'source',
    debut_date: p.filler ? null : `${p.seasons[0]}-04-01`, height_cm: p.filler ? null : 190, weight_kg: p.filler ? null : 90,
    identity_status: 'canonical', aliases: p.slug === 'demo_zoe_arger' ? ['Demo Zoe Arger'] : [],
    clubs: p.clubs.map(clubRef), career_games: careerGames, career_counter_max: careerGames,
    stat_names: [...STAT_COLS], career: statColumns(p.base, careerGames, p.coverage === undefined ? 1 : p.coverage), seasons: seasonLines,
    forecast: forecastByPlayer.get(p.id) ?? null,
    sources: [{ label: 'DEMO fixture generator', url: null, note: DEMO_NOTE }],
    coverage_note: p.coverage === null ? 'DEMO: no per-game statistics recorded for this era.' : p.coverage === 0.5 ? 'DEMO: half the games have statistics recorded.' : 'DEMO: full coverage.',
  };
  put(`players/${p.key}.json`, 'player_detail', detail);
  indexEntries.push({
    id: p.id, key: p.key, name: p.name, clubs: p.clubs.map((c) => club(c).name), first_season: p.seasons[0],
    last_season: p.seasons.at(-1), games: careerGames, active: p.active,
    search: stripDiacritics(`${p.name} ${p.clubs.map((c) => club(c).name).join(' ')}`),
  });
}
put('players/index.json', 'player_index', { count: indexEntries.length, players: indexEntries });

// ---------------------------------------------------------------- predictions
const INTERVAL_ON = { available: true, level: 0.8, method: 'DEMO split-conformal', calibrated: true, reason: null };
const INTERVAL_OFF = { available: false, level: null, method: null, calibrated: false, reason: 'DEMO: no calibrated interval for this set' };
put('predictions/2026/r05.json', 'prediction_set', {
  season: SEASON, stage_id: 'r05', stage_label: 'Demo Round 5', status: 'available', reason: null, units: 'disposals',
  generated_at: GEN, forecast_cutoff: '2026-09-24T00:00:00Z', target_matches: upcoming, model: MODEL, interval: INTERVAL_ON,
  rows: r05Rows,
  omissions: [{ reason: 'insufficient_history', count: 2, detail: 'DEMO omission' }, { reason: 'unresolved_identity', count: 1, detail: null }],
});
put('predictions/2026/qf1.json', 'prediction_set', {
  season: SEASON, stage_id: 'qf1', stage_label: 'Demo Qualifying Final', status: 'expired', reason: 'target matches have been played',
  units: 'disposals', generated_at: '2026-09-01T00:00:00Z', forecast_cutoff: '2026-09-01T00:00:00Z', target_matches: [M2026[6]],
  model: MODEL, interval: INTERVAL_OFF, rows: r04Rows, omissions: [],
});
put('predictions/index.json', 'prediction_index', {
  current: 'predictions/2026/r05.json', status: 'available', reason: null,
  sets: [
    { season: SEASON, stage_id: 'r05', stage_label: 'Demo Round 5', status: 'available', resource: 'predictions/2026/r05.json', rows: r05Rows.length },
    { season: SEASON, stage_id: 'qf1', stage_label: 'Demo Qualifying Final', status: 'expired', resource: 'predictions/2026/qf1.json', rows: r04Rows.length },
  ],
});

// ---------------------------------------------------------------- matches
for (const [season, list] of [[2026, M2026], [2025, M2025], [1994, M1994]]) {
  put(`matches/${season}/index.json`, 'match_index', { season, matches: list });
}
for (const m of ALL_MATCHES) {
  const playersFor = (cid) => PLAYERS.filter((p) => p.clubs.includes(cid) && p.seasons.includes(m.season)).slice(0, 4)
    .map((p) => ({ player_id: p.id, name: p.name, stats: STAT_COLS.map((_, i) => (m.status === 'complete' ? (p.coverage === null ? null : p.base + i) : null)) }));
  const q = (h, a) => ({ quarter: h, home_goals: a?.[0] ?? null, home_behinds: a?.[1] ?? null, away_goals: a?.[2] ?? null, away_behinds: a?.[3] ?? null });
  const done = m.status === 'complete' && m.home.goals !== null;
  put(`matches/detail/${matchKey(m.match_id)}.json`, 'match_detail', {
    summary: m,
    quarters: done ? [q('q1', [1, 1, 1, 1]), q('q2', [2, 2, 2, 2]), q('q3', [3, 3, 3, 3]), q('final', [m.home.goals, m.home.behinds, m.away.goals, m.away.behinds])] : [],
    attendance: done ? 10000 : null,
    home_players: done ? playersFor(m.home.club_id) : [], away_players: done ? playersFor(m.away.club_id) : [],
    stat_columns: STAT_COLS,
    live_snapshots: m.match_id === 'demo:2026:qf1:a-b:replay' ? ['live/demo-live-final/latest.json'] : [],
    sources: [{ label: 'DEMO fixture generator', url: null, note: DEMO_NOTE }],
  });
}

// ---------------------------------------------------------------- teams
const TEAM_SEASONS = [];
function ladderFor(season) {
  const ids = season === 1994 ? ['demo_old', 'demo_a'] : ['demo_a', 'demo_b', 'demo_c', 'demo_d'];
  return ids.map((id, i) => ({
    position: i + 1, club_id: id, name: club(id).name, played: 3, won: 3 - Math.min(i, 3), lost: Math.min(i, 3), drawn: 0,
    points_for: 300 - i * 10, points_against: i === 3 ? 0 : 200 + i * 10, percentage: i === 3 ? null : Math.round(((300 - i * 10) / (200 + i * 10)) * 1000) / 10,
    premiership_points: (3 - Math.min(i, 3)) * 4,
  }));
}
for (const c of CLUBS) {
  const seasons = c.active ? [2025, 2026] : [1994];
  for (const season of seasons) {
    const ladder = ladderFor(season);
    const fixtures = ALL_MATCHES.filter((m) => m.season === season && (m.home.club_id === c.club_id || m.away.club_id === c.club_id));
    const form = fixtures.filter((m) => m.status === 'complete' && m.home.score !== null && m.stage_type === 'regular').map((m) => {
      const home = m.home.club_id === c.club_id;
      const us = home ? m.home.score : m.away.score;
      const them = home ? m.away.score : m.home.score;
      return { match_id: m.match_id, match_date: m.match_date, opponent: home ? m.away.name : m.home.name, result: us > them ? 'W' : us < them ? 'L' : 'D', margin: us - them };
    });
    const leaders = PLAYERS.filter((p) => p.clubs.includes(c.club_id) && p.seasons.includes(season)).slice(0, 3)
      .map((p) => ({ player_id: p.id, name: p.name, stat: 'disposals', value: p.base, observed_games: GAMES_PER_SEASON }));
    const rel = `teams/${c.club_id}/${season}.json`;
    put(rel, 'team_season', {
      club: clubRef(c.club_id), season, ladder, ladder_note: 'DEMO ladder: regular-season matches only; finals excluded.',
      position: ladder.find((r) => r.club_id === c.club_id)?.position ?? null, form_window: 5, form, fixtures,
      team_stats: statLine(100, 3), leaders, five_year: ladder.filter((r) => r.club_id === c.club_id).map((r) => ({ ...r })),
      heuristics: [{ label: 'DEMO heuristic: finals chance', text: 'DEMO: a labelled rule-of-thumb, not a model output.', method: 'DEMO: top-8 by premiership points' }],
    });
    TEAM_SEASONS.push(rel);
  }
}
put('teams/index.json', 'team_index', {
  teams: CLUBS.map((c) => ({
    club_id: c.club_id, name: c.name, lineage_id: c.club_id === 'demo_old' ? 'demo_lineage_old' : `demo_lineage_${c.club_id.slice(-1)}`,
    first_season: c.first, last_season: c.last, active: c.active, seasons: c.active ? [2025, 2026] : [1994],
  })),
});

// ---------------------------------------------------------------- history
const histRows = (scope) => PLAYERS.filter((p) => !p.filler).map((p, i) => ({
  rank: i + 1, player_id: p.id, name: p.name, clubs: p.clubs.map((c) => club(c).name), value: 100 - i * 10,
  value_label: scope === 'career' ? 'DEMO career disposals' : 'DEMO season disposals',
  observed_games: p.coverage === null ? null : 8, eligible_games: 8, coverage: p.coverage === null ? null : p.coverage ?? 1,
  seasons: `${p.seasons[0]}–${p.seasons.at(-1)}`,
}));
const HIST = [
  { category: 'demo_disposals_career', title: 'DEMO career disposals', scope: 'career' },
  { category: 'demo_disposals_season', title: 'DEMO single-season disposals', scope: 'single_season' },
];
const histEntries = [];
for (const h of HIST) {
  const resources = {};
  for (const era of ['all', '1990s', '2020s']) {
    let rows = histRows(h.scope);
    if (era === '1990s') rows = rows.filter((r) => r.seasons.startsWith('19')).map((r, i) => ({ ...r, rank: i + 1 }));
    if (era === '2020s') rows = rows.filter((r) => r.seasons.startsWith('20')).map((r, i) => ({ ...r, rank: i + 1 }));
    const rel = put(`history/${h.category}/${era}.json`, 'history_table', {
      category: h.category, title: h.title, scope: h.scope, era, method: 'DEMO: sum of recorded values', method_version: 'demo-1',
      coverage_note: 'DEMO: rows with missing statistics show "not recorded" coverage.',
      warning: era === '1990s' ? 'DEMO historical-methodology warning: early-era statistics are incomplete.' : null, rows,
    });
    resources[era] = rel;
  }
  histEntries.push({ ...h, eras: Object.keys(resources), resources });
}
put('history/index.json', 'history_index', {
  tables: histEntries,
  era_summary: [{ era: '1990s', players: 2, coverage: 0.25 }, { era: '2020s', players: 64, coverage: 1 }, { era: 'unknown', players: null, coverage: null }],
});

// ---------------------------------------------------------------- accuracy
const mb = (n, mae) => ({ n, mae, rmse: mae === null ? null : mae + 1, bias: mae === null ? null : 0.5, median_ae: mae === null ? null : mae - 0.5, within_5: mae === null ? null : 0.5, within_10: mae === null ? null : 0.75 });
const POP = { intended: 100, predicted: 90, joined: 88, played: 80, missing: 8, excluded: 2, exclusion_reasons: { did_not_play: 8, unresolved_identity: 2 } };
for (const origin of ['prospective', 'replay']) {
  put(`accuracy/demo_model_v1/2026-${origin}.json`, 'accuracy_report', {
    model_id: 'demo_model_v1', baseline_id: 'demo_baseline_mean5', season: SEASON, origin, label: `DEMO ${origin} 2026`,
    headline: mb(80, origin === 'prospective' ? 5 : 4), baseline_headline: mb(80, 6), mean_of_rounds_mae: origin === 'prospective' ? 5.5 : 4.5,
    populations: POP,
    cohorts: [
      { dimension: 'round', cohort: 'Demo Round 1', model: mb(40, 5), baseline: mb(40, 6), sufficient: true },
      { dimension: 'round', cohort: 'Demo Round 2', model: mb(40, 5), baseline: mb(40, 6), sufficient: true },
      { dimension: 'club', cohort: 'Demo Club D', model: mb(3, null), baseline: null, sufficient: false },
    ],
    interval: INTERVAL_ON, interval_coverage: 0.8, interval_median_width: 10, promotion: 'DEMO: champion because demo MAE is lower on demo rows.',
    notes: ['DEMO accuracy report: values are synthetic.'], rows_resource: 'downloads/accuracy-rows.csv',
  });
}
put('accuracy/index.json', 'accuracy_index', {
  champion_model_id: 'demo_model_v1', baseline_model_id: 'demo_baseline_mean5',
  reports: ['prospective', 'replay'].map((origin) => ({ model_id: 'demo_model_v1', season: SEASON, origin, label: `DEMO ${origin} 2026`, resource: `accuracy/demo_model_v1/2026-${origin}.json` })),
  model_card: 'DEMO model card: a synthetic model used to exercise the accuracy page. Units: disposals per player-game.',
});

// ---------------------------------------------------------------- lists
put('lists/2026.json', 'lists_season', {
  season: SEASON,
  drafts: [
    { season: SEASON, event_type: 'national_draft', draft_round: 1, pick: 1, club: 'Demo Club A', player_name: 'Demo Draftee One', player_id: null, recruited_from: 'Demo Academy', grade: null },
    { season: SEASON, event_type: 'rookie_draft', draft_round: 1, pick: 2, club: 'Demo Club B', player_name: 'Demo Draftee Two', player_id: null, recruited_from: null, grade: 'DEMO grade' },
  ],
  contracts: [{ player_name: 'Demo Player A1', player_id: 'legacy:demo_player_a1', club: 'Demo Club A', contract_end: 2028, fa_category: null, observed_at: '2026-09-01', source_type: 'DEMO observation', notes: 'DEMO: observation, not a guaranteed current contract.' }],
  schools: [{ draft_year: 2026, pick: 1, player_name: 'Demo Draftee One', player_id: null, school: 'Demo School', school_type: 'DEMO', confidence: 'low' }],
  source_note: 'DEMO: fixture fallback list observed 2026-09-01.', sources: [{ label: 'DEMO fixture generator', url: null, note: DEMO_NOTE }],
});
put('lists/index.json', 'lists_index', { seasons: [SEASON], resources: { '2026': 'lists/2026.json' } });

// ---------------------------------------------------------------- articles
const ARTICLES = [
  { slug: 'demo-article-one', title: 'DEMO article one: reading the demo forecast', category: 'analysis', published: '2026-09-20', scope: 'frozen', state: 'published_archive' },
  { slug: 'demo-article-two', title: 'DEMO article two: demo round recap', category: 'recap', published: '2026-03-30', scope: 'live', state: 'generated' },
  { slug: 'demo-article-archive', title: 'DEMO archive article from a previous season', category: 'archive', published: '2025-04-01', scope: 'archive', state: 'published_archive' },
];
const summaries = ARTICLES.map((a) => ({
  slug: a.slug, title: a.title, category: a.category, published: a.published, as_of: `DEMO as of ${a.published}`, scope: a.scope,
  excerpt: `DEMO excerpt for ${a.title}.`, editorial_state: a.state, original_path: `docs/demo/${a.slug}.md`, resource: `articles/${a.slug}.json`,
}));
for (const s of summaries) {
  put(s.resource, 'article', {
    summary: s,
    html: `<h2>DEMO section</h2><p>This is <strong>demo</strong> article text for <em>${s.title.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</em>. It contains no real statistics.</p><ul><li>DEMO point one</li><li>DEMO point two</li></ul><p><a href="../demo-article-one/">Related DEMO article</a></p><table><caption>DEMO table</caption><thead><tr><th scope="col">Label</th><th scope="col">Value</th></tr></thead><tbody><tr><td>DEMO</td><td>1</td></tr></tbody></table>`,
    sources: [{ label: 'DEMO fixture generator', url: null, note: DEMO_NOTE }], provenance: 'DEMO: generated by web/scripts/make-demo-fixture.mjs',
  });
}
put('articles/index.json', 'article_index', { articles: summaries });

// ---------------------------------------------------------------- live
const liveSnap = (id, final, fetched) => ({
  source_game_id: id, match_id: final ? 'demo:2026:qf1:a-b:replay' : 'demo:2026:r05:a-c', fetched_at: fetched, status: final ? 'final' : 'in_progress',
  quarter: final ? 'final' : 'q2', home: team('demo_a', final ? 11 : 3, final ? 11 : 2), away: team(final ? 'demo_b' : 'demo_c', final ? 10 : 2, final ? 10 : 4),
  reliable_fields: ['disposals', 'kicks', 'handballs'], unavailable_fields: ['goals', 'behinds', 'clangers'],
  players: [{ player_id: 'legacy:demo_player_a1', name: 'Demo Player A1', stats: { disposals: 10, kicks: 6, handballs: 4 } }],
  timeline: [{ quarter: 'q1', event: 'DEMO quarter break', home_score: 20, away_score: 16 }, { quarter: 'q2', event: 'DEMO snapshot', home_score: null, away_score: null }],
  reads: ['DEMO deterministic read: leading by a demo margin.'], anomalies: final ? [] : ['DEMO anomaly: unnamed column ignored'], final,
});
put('live/demo-live-final/latest.json', 'live_snapshot', liveSnap('demo-live-final', true, '2026-09-12T12:30:00Z'));
put('live/demo-live-progress/latest.json', 'live_snapshot', liveSnap('demo-live-progress', false, '2026-09-24T23:50:00Z'));
put('live/index.json', 'live_index', {
  delivery: 'DEMO: snapshot archive (static host); not a real-time feed.',
  matches: [
    { source_game_id: 'demo-live-final', match_id: 'demo:2026:qf1:a-b:replay', label: 'DEMO: Demo Club A v Demo Club B (replay)', last_fetched_at: '2026-09-12T12:30:00Z', final: true, resource: 'live/demo-live-final/latest.json' },
    { source_game_id: 'demo-live-progress', match_id: 'demo:2026:r05:a-c', label: 'DEMO: Demo Club A v Demo Club C', last_fetched_at: '2026-09-24T23:50:00Z', final: false, resource: 'live/demo-live-progress/latest.json' },
  ],
});

// ---------------------------------------------------------------- quality
put('quality.json', 'quality', {
  dataset_status: 'demo', table_counts: { players: PLAYERS.length, matches: ALL_MATCHES.length, player_games: PLAYERS.length * GAMES_PER_SEASON },
  quarantined_rows: 0,
  issues: [
    { rule_id: 'DEMO-001', severity: 'info', count: 1, description: 'DEMO issue: one demo player has no recorded statistics.' },
    { rule_id: 'DEMO-002', severity: 'warning', count: 1, description: 'DEMO issue: one demo match has no recorded score.' },
  ],
  limitations: ['DEMO: this release is synthetic and exists only to test the interface.'],
  sources: [{ label: 'DEMO fixture generator', url: null, note: DEMO_NOTE }],
});

// ---------------------------------------------------------------- downloads (real files with real hashes)
function csv(rows) {
  return rows.map((r) => r.map((v) => {
    const s = v === null ? '' : String(v);
    const safe = /^[=+\-@\t\r]/.test(s) && !/^-?\d/.test(s) ? `'${s}` : s;
    return /[",\n]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
  }).join(',')).join('\n') + '\n';
}
const playersCsv = csv([['player_id', 'name', 'clubs', 'first_season', 'last_season', 'games'], ...indexEntries.map((e) => [e.id, e.name, e.clubs.join('; '), e.first_season, e.last_season, e.games])]);
putRaw('downloads/players.csv', playersCsv);
const predCsv = csv([['player_id', 'player_name', 'stage_label', 'predicted_disposals', 'interval_low', 'interval_high'], ...r05Rows.map((r) => [r.player_id, r.player_name, r.stage_label, r.predicted_disposals, r.interval_low, r.interval_high])]);
putRaw('downloads/predictions.csv', predCsv);
putRaw('downloads/accuracy-rows.csv', csv([['prediction_id', 'origin', 'abs_error'], ['demo-pred-1', 'prospective', 5], ['demo-pred-2', 'prospective', 5]]));
const readme = `# DEMO fan pack\n\n${DEMO_NOTE}\n\nRelease: ${RELEASE_ID}\n`;
putRaw('downloads/README.md', readme);
function storedZip(files) {
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const [name, content] of files) {
    const data = Buffer.from(content);
    const nameBuf = Buffer.from(name);
    const crc = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0, 6); local.writeUInt16LE(0, 8);
    local.writeUInt16LE(0, 10); local.writeUInt16LE(0x21, 12); // 1980-01-01 00:00 (deterministic)
    local.writeUInt32LE(crc, 14); local.writeUInt32LE(data.length, 18); local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26); local.writeUInt16LE(0, 28);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(0, 8);
    central.writeUInt16LE(0, 10); central.writeUInt16LE(0, 12); central.writeUInt16LE(0x21, 14); central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20); central.writeUInt32LE(data.length, 24); central.writeUInt16LE(nameBuf.length, 28);
    central.writeUInt32LE(offset, 42);
    locals.push(local, nameBuf, data);
    centrals.push(central, nameBuf);
    offset += 30 + nameBuf.length + data.length;
  }
  const cd = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(files.length, 8); end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, end]);
}
putRaw('downloads/fan-pack.zip', storedZip([['README.md', readme], ['players.csv', playersCsv], ['predictions.csv', predCsv]]));
const dl = (key, label, kind, path, rows = null) => ({ key, label, kind, path, bytes: written.get(path).length, sha256: sha(written.get(path)), as_of: 'DEMO 2026-09-25', rows });
put('downloads.json', 'downloads', {
  release_id: RELEASE_ID,
  items: [
    dl('players_csv', 'DEMO player directory (all rows)', 'csv', 'downloads/players.csv', indexEntries.length),
    dl('predictions_csv', 'DEMO current predictions (all rows)', 'csv', 'downloads/predictions.csv', r05Rows.length),
    dl('accuracy_rows_csv', 'DEMO scored accuracy rows (all rows)', 'csv', 'downloads/accuracy-rows.csv', 2),
    dl('fan_pack', 'DEMO fan pack ZIP', 'zip', 'downloads/fan-pack.zip'),
    dl('readme', 'DEMO release README', 'md', 'downloads/README.md'),
  ],
  retained: [],
});

// ---------------------------------------------------------------- overview
const leaders = PLAYERS.filter((p) => p.active).slice(0, 5).map((p) => ({ player_id: p.id, name: p.name, stat: 'disposals', value: p.base, observed_games: 4 }));
put('overview.json', 'overview', {
  release_id: RELEASE_ID, snapshot_id: SNAPSHOT_ID, season: SEASON, demo: true,
  freshness: {
    source_checked_at: '2026-09-24T22:00:00Z', latest_completed_match_at: '2026-09-12T09:40:00Z', latest_completed_match_date: '2026-09-12',
    coverage_through: 'DEMO Qualifying Final (replay)', generated_at: GEN, published_at: null, validation_state: 'PASS',
    dataset_status: 'demo', season_active: true, stale: false, stale_reason: null,
  },
  next_fixture_status: 'available', next_fixture_reason: null, upcoming, prediction_highlights: r05Rows.slice(0, 3),
  form_highlights: leaders.slice(0, 3), recent_results: recent, latest_articles: summaries.slice(0, 2), leaders,
  model_status: { forecast_status: 'available', reason: null, model: MODEL },
  warnings: ['DEMO release: every name and number on this site is synthetic.'],
});

// ---------------------------------------------------------------- manifest
const resources = {
  overview: ref('overview.json'), prediction_index: ref('predictions/index.json'), player_index: ref('players/index.json'),
  team_index: ref('teams/index.json'), 'match_index:2026': ref('matches/2026/index.json'), 'match_index:2025': ref('matches/2025/index.json'),
  'match_index:1994': ref('matches/1994/index.json'), history_index: ref('history/index.json'), accuracy_index: ref('accuracy/index.json'),
  lists_index: ref('lists/index.json'), article_index: ref('articles/index.json'), live_index: ref('live/index.json'),
  quality: ref('quality.json'), downloads: ref('downloads.json'),
};
put('release.json', 'release', {
  schema_version: 1, release_id: RELEASE_ID, snapshot_id: SNAPSHOT_ID, generated_at: GEN, season: SEASON, demo: true, base_label: 'DEMO',
  coverage: { status: 'demo', through: 'DEMO Qualifying Final (replay)' },
  forecast: { status: 'available', reason: null, artifact: 'predictions/2026/r05.json', model_id: 'demo_model_v1' }, resources,
});

rmSync(OUT, { recursive: true, force: true });
for (const [rel, bytes] of written) {
  const p = join(OUT, rel);
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, bytes);
}
console.log(`wrote ${written.size} files to ${OUT}`);
