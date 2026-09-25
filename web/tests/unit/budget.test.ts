// scripts/budget-lib.mjs: payload budgets measured on the production build (PLAN 12).
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { gzipSync } from 'node:zlib';
import {
  BUDGETS, evaluate, gzipSize, htmlAssets, measureSite, staticImports,
} from '../../scripts/budget-lib.mjs';

let dir: string;
const put = (rel: string, body: string | Buffer) => {
  const p = join(dir, rel);
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, body);
};

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'scvia-budget-'));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

describe('parsing', () => {
  it('finds stylesheets, module scripts, preloads and island component/renderer URLs', () => {
    const html = `<html data-release="r1" data-base="/sub/"><head>
      <link rel="icon" href="/sub/favicon.svg"><link rel="stylesheet" href="/sub/_astro/a.css">
      <link rel="modulepreload" href="/sub/_astro/pre.js"><script type="module" src="/sub/_astro/main.js"></script>
      <script>inline()</script></head><body>
      <astro-island component-url="/sub/_astro/Island.js" renderer-url="/sub/_astro/client.js" props="{}"></astro-island>
      <astro-island component-url="/sub/_astro/Island.js" renderer-url="/sub/_astro/client.js"></astro-island></body></html>`;
    const a = htmlAssets(html);
    expect(a.css).toEqual(['/sub/_astro/a.css']);
    expect(a.js.sort()).toEqual(['/sub/_astro/Island.js', '/sub/_astro/client.js', '/sub/_astro/main.js', '/sub/_astro/pre.js']);
    expect(a.releaseId).toBe('r1');
    expect(a.base).toBe('/sub/');
  });
  it('follows static imports and re-exports but not dynamic import()', () => {
    const js = 'import{t as e}from"./react.js";import"./side.js";export{x}from"./re.js";const v=()=>import("./lazy.js");';
    expect(staticImports(js).sort()).toEqual(['./re.js', './react.js', './side.js']);
  });
  it('measures gzip size of the bytes', () => {
    const b = Buffer.from('x'.repeat(10000));
    expect(gzipSize(b)).toBe(gzipSync(b, { level: 6 }).length);
    expect(gzipSize(b)).toBeLessThan(200);
  });
});

function site() {
  put('index.html', '<html data-release="r1" data-base="/"><link rel="stylesheet" href="/_astro/s.css"><script type="module" src="/_astro/main.js"></script></html>');
  put('player/index.html', '<html data-release="r1" data-base="/"><astro-island component-url="/_astro/P.js" renderer-url="/_astro/client.js"></astro-island></html>');
  put('_astro/s.css', 'body{color:red}');
  put('_astro/main.js', 'import"./shared.js";console.log(1)');
  put('_astro/P.js', 'import{a}from"./shared.js";const l=()=>import("./player_detail.generated.X1.js");');
  put('_astro/client.js', 'export const c=1');
  put('_astro/shared.js', 'export const a=1');
  put('_astro/player_detail.generated.X1.js', 'import"./formats.js";export const validate=1');
  put('_astro/release.generated.X2.js', 'export const validate=2');
  put('_astro/formats.js', 'export const f=1');
  put('data/r1/release.json', '{"release_id":"r1"}');
  put('data/r1/players/index.json', JSON.stringify({ players: Array.from({ length: 50 }, (_, i) => ({ id: `p${i}` })) }));
  put('data/r1/players/small.json', '{"id":"s"}');
  put('data/r1/players/big.json', JSON.stringify({ id: 'b', pad: 'y'.repeat(5000) }));
  put('data/r1/matches/detail/m1.json', '{"m":1}');
}

const ROUTES = [
  { route: '', json: [] },
  { route: 'player/', json: [{ kind: 'release', glob: 'release.json' }, { kind: 'player_detail', glob: 'players/*.json', exclude: ['players/index.json'] }] },
];

describe('measureSite', () => {
  it('counts HTML, CSS, JS closure, lazily-validated kinds and the worst-case initial JSON per route', () => {
    site();
    const r = measureSite(dir, { routes: ROUTES });
    const home = r.routes.find((x) => x.route === '')!;
    expect(home.js_files.sort()).toEqual(['_astro/main.js', '_astro/shared.js']);
    expect(home.css_files).toEqual(['_astro/s.css']);
    expect(home.json_files).toEqual([]);
    const player = r.routes.find((x) => x.route === 'player/')!;
    // island + renderer + static imports + validators for the kinds it loads (+ their imports); never the dynamic import target by itself
    expect(player.js_files.sort()).toEqual([
      '_astro/P.js', '_astro/client.js', '_astro/formats.js', '_astro/player_detail.generated.X1.js', '_astro/release.generated.X2.js', '_astro/shared.js',
    ]);
    expect(player.json_files).toEqual(['data/r1/release.json', 'data/r1/players/big.json']);
    expect(player.total_gzip).toBe(player.html_gzip + player.css_gzip + player.js_gzip + player.json_gzip);
    expect(r.player_index.path).toBe('data/r1/players/index.json');
    expect(r.largest_detail.path).toBe('data/r1/players/big.json');
    expect(r.release_id).toBe('r1');
  });
  it('lists every release JSON (largest first) and the total artifact bytes', () => {
    site();
    const r = measureSite(dir, { routes: ROUTES });
    expect(r.data_json.map((d: { path: string }) => d.path)).toContain('data/r1/matches/detail/m1.json');
    expect(r.data_json).toHaveLength(5);
    expect(r.data_json[0].gzip).toBeGreaterThanOrEqual(r.data_json[1].gzip);
    expect(r.artifact_bytes).toBeGreaterThan(5000);
  });
  it('fails loudly when a referenced asset is missing from the build', () => {
    site();
    rmSync(join(dir, '_astro/shared.js'));
    expect(() => measureSite(dir, { routes: ROUTES })).toThrow(/missing.*shared\.js/);
  });
  it('fails when a route page was not built', () => {
    site();
    expect(() => measureSite(dir, { routes: [{ route: 'nope/', json: [] }] })).toThrow(/nope/);
  });
});

describe('evaluate', () => {
  const measured = (o: Partial<{ total: number; js: number; idx: number; detail: number; other: number; bytes: number }>) => ({
    release_id: 'r1', base: '/',
    routes: [{ route: 'x/', core: true, html_gzip: 0, css_gzip: 0, js_gzip: o.js ?? 1000, json_gzip: 0, total_gzip: o.total ?? 1000, js_files: [], css_files: [], json_files: [] }],
    player_index: { path: 'i', gzip: o.idx ?? 1000 },
    largest_detail: { path: 'd', gzip: o.detail ?? 1000 },
    data_json: [{ path: 'data/r1/players/index.json', gzip: o.idx ?? 1000 }, { path: 'data/r1/history/x/all.json', gzip: o.other ?? 1000 }, { path: 'd', gzip: o.detail ?? 1000 }],
    artifact_bytes: o.bytes ?? 1000,
  });
  it('passes within budget', () => {
    const r = evaluate(measured({}));
    expect(r.pass).toBe(true);
    expect(r.failures).toEqual([]);
  });
  it('fails each budget independently and names it', () => {
    expect(evaluate(measured({ total: BUDGETS.initial_route_gzip + 1 })).failures[0]).toMatch(/x\/.*initial route/);
    expect(evaluate(measured({ js: BUDGETS.route_js_gzip + 1 })).failures[0]).toMatch(/x\/.*JavaScript/);
    expect(evaluate(measured({ idx: BUDGETS.player_index_gzip + 1 })).failures[0]).toMatch(/player index/);
    expect(evaluate(measured({ detail: BUDGETS.detail_json_gzip + 1 })).failures[0]).toMatch(/JSON d .*>/);
    expect(evaluate(measured({ bytes: BUDGETS.artifact_bytes + 1 })).failures[0]).toMatch(/artifact/);
  });
  it('holds every release JSON except the player index to the per-file limit (split or paginate if larger)', () => {
    const r = evaluate(measured({ other: BUDGETS.detail_json_gzip + 1 }));
    expect(r.failures).toEqual([expect.stringMatching(/history\/x\/all\.json/)]);
    expect(evaluate(measured({ idx: BUDGETS.detail_json_gzip + 1 })).failures).toEqual([]);
  });
  it('treats the exact budget as within budget', () => {
    expect(evaluate(measured({ total: BUDGETS.initial_route_gzip, js: BUDGETS.route_js_gzip })).pass).toBe(true);
  });
  it('uses the PLAN section 12 limits in KiB', () => {
    expect(BUDGETS).toEqual({
      initial_route_gzip: 250 * 1024, route_js_gzip: 120 * 1024, player_index_gzip: 750 * 1024, detail_json_gzip: 150 * 1024, artifact_bytes: 300 * 1024 * 1024,
    });
  });
});
