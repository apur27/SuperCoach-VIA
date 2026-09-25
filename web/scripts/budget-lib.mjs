// Payload budgets (PLAN section 12) measured on a production build directory.
// Everything is computed from built bytes; nothing is estimated.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, posix } from 'node:path';
import { gzipSync } from 'node:zlib';

const KiB = 1024;
export const BUDGETS = {
  initial_route_gzip: 250 * KiB, // HTML + CSS + JS + initial JSON, per core route
  route_js_gzip: 120 * KiB, // JavaScript per core route
  player_index_gzip: 750 * KiB, // players/index.json (loaded on search interaction only)
  detail_json_gzip: 150 * KiB, // any single release JSON other than the player index; split or paginate if larger
  artifact_bytes: 300 * 1024 * KiB, // whole built site, uncompressed
};
export const GZIP_LEVEL = 6;

export function gzipSize(buf) {
  return gzipSync(buf, { level: GZIP_LEVEL }).length;
}

const attr = (tag, name) => new RegExp(`\\s${name}="([^"]*)"`).exec(tag)?.[1];

/** Assets an HTML page requests up front: stylesheets, module scripts/preloads, island component + renderer. */
export function htmlAssets(html) {
  const css = new Set();
  const js = new Set();
  for (const [tag] of html.matchAll(/<link\b[^>]*>/g)) {
    const rel = attr(tag, 'rel');
    const href = attr(tag, 'href');
    if (!href) continue;
    if (rel === 'stylesheet') css.add(href);
    if (rel === 'modulepreload') js.add(href);
  }
  for (const [tag] of html.matchAll(/<script\b[^>]*>/g)) {
    const src = attr(tag, 'src');
    if (src) js.add(src);
  }
  for (const [tag] of html.matchAll(/<astro-island\b[^>]*>/g)) {
    for (const name of ['component-url', 'renderer-url', 'before-hydration-url']) {
      const u = attr(tag, name);
      if (u) js.add(u);
    }
  }
  const htmlTag = /<html\b[^>]*>/.exec(html)?.[0] ?? '';
  return { css: [...css], js: [...js], releaseId: attr(htmlTag, 'data-release') ?? null, base: attr(htmlTag, 'data-base') ?? '/' };
}

/** Static import / re-export specifiers of a built ES module (dynamic import() excluded). */
export function staticImports(code) {
  const out = new Set();
  const re = /(?:^|[;\s}])(?:import|export)\s*(?:[\w$*{}\s,]*?\bfrom\s*)?["']([^"']+)["']/g;
  for (const m of code.matchAll(re)) out.add(m[1]);
  return [...out];
}

function segmentRe(seg) {
  return new RegExp(`^${seg.split('*').map((s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('[^/]*')}$`);
}

/** Files under root matching a simple glob ("players/*.json", "player-games/*\/*.json"). */
export function globFiles(root, pattern) {
  const segs = pattern.split('/');
  let paths = [''];
  for (const [i, seg] of segs.entries()) {
    const re = segmentRe(seg);
    const last = i === segs.length - 1;
    paths = paths.flatMap((p) => {
      const abs = join(root, p);
      if (!existsSync(abs) || !statSync(abs).isDirectory()) return [];
      return readdirSync(abs).sort().filter((n) => re.test(n)).map((n) => (p ? `${p}/${n}` : n))
        .filter((r) => (last ? statSync(join(root, r)).isFile() : statSync(join(root, r)).isDirectory()));
    });
  }
  return paths;
}

function walkFiles(root, rel = '') {
  return readdirSync(join(root, rel)).sort().flatMap((n) => {
    const r = rel ? `${rel}/${n}` : n;
    return statSync(join(root, r)).isDirectory() ? walkFiles(root, r) : [r];
  });
}

function readRequired(file, what) {
  if (!existsSync(file) || !statSync(file).isFile()) throw new Error(`missing ${what}: ${file}`);
  return readFileSync(file);
}

/**
 * Measure a built site. `routes`: [{ route, core?, json: [{ kind, glob, exclude?, count? }] }].
 * For each JSON entry the `count` LARGEST matching files (by gzip) are charged to the route:
 * a worst case over the release, independent of which ID a visitor opens.
 */
export function measureSite(dist, { routes }) {
  const astroDir = join(dist, '_astro');
  const astroFiles = existsSync(astroDir) ? readdirSync(astroDir) : [];
  const sizeCache = new Map();
  const gz = (rel, what) => {
    if (!sizeCache.has(rel)) sizeCache.set(rel, gzipSize(readRequired(join(dist, rel), what)));
    return sizeCache.get(rel);
  };
  let releaseId = null;
  let base = '/';
  const results = [];
  for (const spec of routes) {
    const htmlRel = `${spec.route}index.html`;
    const html = readRequired(join(dist, htmlRel), `route page for "${spec.route}"`).toString('utf8');
    const assets = htmlAssets(html);
    releaseId ??= assets.releaseId;
    base = assets.base;
    const toRel = (url) => {
      const p = url.split(/[?#]/)[0];
      if (!p.startsWith(base)) throw new Error(`asset outside base ${base}: ${url}`);
      return p.slice(base.length);
    };
    const js = new Set();
    const visit = (rel) => {
      if (js.has(rel)) return;
      const code = readRequired(join(dist, rel), `JavaScript asset ${rel}`).toString('utf8');
      js.add(rel);
      for (const spec2 of staticImports(code)) {
        if (!spec2.startsWith('.') && !spec2.startsWith('/')) continue;
        visit(spec2.startsWith('/') ? toRel(spec2) : posix.normalize(posix.join(posix.dirname(rel), spec2)));
      }
    };
    for (const u of assets.js) visit(toRel(u));
    const jsonFiles = [];
    for (const entry of spec.json) {
      // Validators are code-split per resource kind and fetched with the resource.
      const v = astroFiles.find((n) => new RegExp(`^${entry.kind}\\.generated\\.[^.]+\\.js$`).test(n));
      if (!v) throw new Error(`missing validator chunk for kind ${entry.kind} in ${astroDir}`);
      visit(`_astro/${v}`);
      const relRoot = `data/${assets.releaseId}`;
      const candidates = globFiles(join(dist, relRoot), entry.glob).filter((f) => !(entry.exclude ?? []).includes(f))
        .map((f) => `${relRoot}/${f}`);
      if (!candidates.length) throw new Error(`route "${spec.route}": no JSON matches ${entry.glob}`);
      candidates.sort((a, b) => gz(b) - gz(a) || a.localeCompare(b));
      jsonFiles.push(...candidates.slice(0, entry.count ?? 1));
    }
    const css = assets.css.map(toRel);
    const sum = (files) => files.reduce((t, f) => t + gz(f), 0);
    const parts = { html_gzip: gz(htmlRel), css_gzip: sum(css), js_gzip: sum([...js]), json_gzip: sum(jsonFiles) };
    results.push({
      route: spec.route, core: spec.core ?? true, ...parts,
      total_gzip: parts.html_gzip + parts.css_gzip + parts.js_gzip + parts.json_gzip,
      js_files: [...js], css_files: css, json_files: jsonFiles,
    });
  }
  const relRoot = `data/${releaseId}`;
  const idxRel = `${relRoot}/players/index.json`;
  const files = walkFiles(dist);
  const dataJson = files.filter((f) => f.startsWith(`${relRoot}/`) && f.endsWith('.json'))
    .map((f) => ({ path: f, gzip: gz(f) }))
    .sort((a, b) => b.gzip - a.gzip || a.path.localeCompare(b.path));
  return {
    release_id: releaseId, base, gzip: `zlib level ${GZIP_LEVEL}`,
    routes: results,
    player_index: { path: idxRel, gzip: gz(idxRel, 'player index') },
    largest_detail: dataJson.find((d) => d.path !== idxRel) ?? { path: null, gzip: 0 },
    data_json: dataJson,
    artifact_bytes: files.reduce((n, f) => n + statSync(join(dist, f)).size, 0),
  };
}

/** Compare measurements with BUDGETS (a value equal to the budget passes). */
export function evaluate(m, budgets = BUDGETS) {
  const failures = [];
  const kib = (n) => `${(n / KiB).toFixed(1)} KiB`;
  for (const r of m.routes.filter((x) => x.core)) {
    if (r.total_gzip > budgets.initial_route_gzip) failures.push(`/${r.route}: initial route transfer ${kib(r.total_gzip)} > ${kib(budgets.initial_route_gzip)}`);
    if (r.js_gzip > budgets.route_js_gzip) failures.push(`/${r.route}: JavaScript ${kib(r.js_gzip)} > ${kib(budgets.route_js_gzip)}`);
  }
  if (m.player_index.gzip > budgets.player_index_gzip) failures.push(`player index ${kib(m.player_index.gzip)} > ${kib(budgets.player_index_gzip)}`);
  for (const d of m.data_json) {
    if (!d.path.endsWith('/players/index.json') && d.gzip > budgets.detail_json_gzip) failures.push(`JSON ${d.path} ${kib(d.gzip)} > ${kib(budgets.detail_json_gzip)}`);
  }
  if (m.artifact_bytes > budgets.artifact_bytes) failures.push(`artifact ${(m.artifact_bytes / 1048576).toFixed(1)} MiB > ${budgets.artifact_bytes / 1048576} MiB`);
  return { ...m, budgets, failures, pass: failures.length === 0 };
}
