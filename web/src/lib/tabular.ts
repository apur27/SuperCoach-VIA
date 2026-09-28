/** Parse a published CSV fact table and project the columns a page shows. */

export const BROWNLOW_VIEW_COLUMNS = [
  'rank', 'name', 'club', 'games', 'proxy_per_game', 'season_proxy_scaled', 'ineligible', 'label',
] as const;

export const ERA_VIEW_COLUMNS = [
  'era', 'metric', 'n_player_games', 'n_with_metric', 'mean_per_game', 'recording_status',
] as const;

export interface CsvTable {
  columns: string[];
  rows: string[][];
}

export function parseCsv(text: string): CsvTable {
  const records = splitRecords(text.replace(/^\uFEFF/, '').trim());
  const header = records[0];
  if (!header) return { columns: [], rows: [] };
  return { columns: header, rows: records.slice(1) };
}

export function project(table: CsvTable, want: readonly string[]): CsvTable {
  const missing = want.filter((c) => !table.columns.includes(c));
  if (missing.length) throw new Error(`missing columns: ${missing.join(', ')}`);
  const idx = want.map((c) => table.columns.indexOf(c));
  return { columns: [...want], rows: table.rows.map((r) => idx.map((i) => r[i] ?? '')) };
}

function splitRecords(text: string): string[][] {
  if (!text) return [];
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; }
        else quoted = false;
      } else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell);
      cell = '';
      rows.push(row);
      row = [];
    } else cell += c;
  }
  if (cell !== '' || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.some((x) => x !== ''));
}
