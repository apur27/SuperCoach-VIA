import { describe, expect, it } from 'vitest';
import { BROWNLOW_VIEW_COLUMNS, ERA_VIEW_COLUMNS, parseCsv, project } from '../../src/lib/tabular';

describe('exported fact tables', () => {
  it('parses quoted fields and keeps the proxy label', () => {
    const text = [
      'rank,name,club,games,proxy_per_game,label',
      '1,"Citizen, A",Lions,22,1.25,Brownlow proxy (not votes)',
    ].join('\n');
    const table = parseCsv(text);
    const view = project(table, ['rank', 'name', 'proxy_per_game', 'label']);
    expect(view.rows[0]).toEqual(['1', 'Citizen, A', '1.25', 'Brownlow proxy (not votes)']);
  });

  it('refuses a Brownlow export that dropped the proxy label', () => {
    const table = parseCsv('rank,name\n1,A\n');
    expect(() => project(table, BROWNLOW_VIEW_COLUMNS)).toThrow(/label/);
  });

  it('projects the era summary columns used on the history page', () => {
    const text = 'era,metric,n_player_games,n_with_metric,mean_per_game,recording_status,extra\n1990s,disposals,10,4,12.5,partial,ignore\n';
    const view = project(parseCsv(text), ERA_VIEW_COLUMNS);
    expect(view.columns).toEqual([...ERA_VIEW_COLUMNS]);
    expect(view.rows).toEqual([['1990s', 'disposals', '10', '4', '12.5', 'partial']]);
  });
});
