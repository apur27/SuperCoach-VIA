// Route -> initial JSON map for scripts/budget.mjs. tests/e2e/budget-map.spec.ts checks it
// against the requests the production build actually makes on first load.
const R = { kind: 'release', glob: 'release.json' };
const PLAYER = { kind: 'player_detail', glob: 'players/*.json', exclude: ['players/index.json'] };
// Initial JSON each route fetches on load. Detail routes are charged the LARGEST matching
// file(s) in the release, so the budget holds for whichever ID a visitor opens.
export const ROUTES = [
  { route: '', json: [] },
  { route: 'predictions/', json: [R, { kind: 'prediction_index', glob: 'predictions/index.json' }, { kind: 'prediction_set', glob: 'predictions/*/*.json', optional: true }] },
  { route: 'players/', json: [] }, // index loads only on search interaction (measured separately)
  { route: 'player/', json: [R, PLAYER, { kind: 'player_season_games', glob: 'player-games/*/*.json' }] },
  { route: 'compare/', json: [R, { ...PLAYER, count: 4 }] },
  { route: 'teams/', json: [] },
  { route: 'team/', json: [R, { kind: 'team_index', glob: 'teams/index.json' }, { kind: 'team_season', glob: 'teams/*/*.json' }] },
  { route: 'matches/', json: [R, { kind: 'match_index', glob: 'matches/*/index.json' }] },
  { route: 'match/', json: [R, { kind: 'match_detail', glob: 'matches/detail/*.json' }] },
  { route: 'history/', json: [R, { kind: 'history_table', glob: 'history/*/*.json' }] },
  { route: 'accuracy/', json: [R, { kind: 'accuracy_report', glob: 'accuracy/*/*.json', optional: true }] },
  { route: 'articles/', json: [] },
  { route: 'lists/', core: false, json: [R, { kind: 'lists_season', glob: 'lists/*.json', exclude: ['lists/index.json'] }] },
  { route: 'live/', core: false, json: [R, { kind: 'live_snapshot', glob: 'live/*/*.json', optional: true }] },
  { route: 'watchlist/', core: false, json: [] },
  { route: 'downloads/', core: false, json: [] },
  { route: 'data-status/', core: false, json: [] },
  { route: 'methodology/', core: false, json: [] },
];
