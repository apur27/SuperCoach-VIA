// Deterministic payload budgets (PLAN §12) over a production build.
// Usage: node scripts/budget.mjs [--dir dist] [--json out.json]
// Per route: HTML + CSS + JS (entry scripts and their static imports, transitively) +
// initial JSON the route fetches on load (overview for "/"), all gzip-compressed.
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, posix, relative, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';

export const BUDGETS = {
  routeTotalGzip: 250 * 1024,
  routeJsGzip: 120 * 1024,
  playerIndexGzip: 750 * 1024,
  detailJsonGzip: 150 * 1024,
};

const gz = (buf) => gzipSync(buf, { level: 9 }).length;

function walk(dir) {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

/** Strip the configured base so "/SuperCoach-VIA/_astro/x.js" maps to dist/_astro/x.js. */
function assetPath(dist, url, base) {
  if (!url.startsWith(base)) return null;
  const p = join(dist, url.slice(base.length));
  return existsSync(p) && statSync(p).isFile() ? p : null;
}

function jsClosure(dist, entry, seen) {
  if (seen.has(entry)) return;
  seen.add(entry);
  const code = readFileSync(entry, 'utf8');
  // Static imports only; dynamic import() chunks load on interaction and are not initial transfer.
  for (const m of code.matchAll(/(?:^|[;\s}])import\s*(?:[\w*{}\s,$]+from\s*)?["']([^"']+)["']/g)) {
    const spec = m[1];
    if (!spec.startsWith('.')) continue;
    const p = resolve(posix.dirname(entry), spec);
    if (existsSync(p)) jsClosure(dist, p, seen);
  }
}

export function measure(dist, base = '/') {
  const routes = [];
  const htmlFiles = walk(dist).filter((f) => f.endsWith('.html'));
  const releaseDirs = existsSync(join(dist, 'data')) ? readdirSync(join(dist, 'data')) : [];
  for (const html of htmlFiles) {
    const text = readFileSync(html, 'utf8');
    const css = new Set();
    const js = new Set();
    for (const m of text.matchAll(/<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"|<link[^>]+href="([^"]+)"[^>]+rel="stylesheet"/g)) {
      const p = assetPath(dist, m[1] ?? m[2], base);
      if (p) css.add(p);
    }
    for (const m of text.matchAll(/(?:<script[^>]+src|component-url|renderer-url)="([^"]+)"/g)) {
      const p = assetPath(dist, m[1], base);
      if (p) jsClosure(dist, p, js);
    }
    const route = '/' + relative(dist, html).replace(/index\.html$/, '').replaceAll('\\', '/');
    const json = [];
    if (route === '/') {
      for (const rel of releaseDirs) {
        const p = join(dist, 'data', rel, 'overview.json');
        if (existsSync(p)) json.push(p);
      }
    }
    const size = (files) => [...files].reduce((n, f) => n + gz(readFileSync(f)), 0);
    const htmlGz = gz(readFileSync(html));
    const cssGz = size(css);
    const jsGz = size(js);
    const jsonGz = size(json);
    routes.push({ route, html: htmlGz, css: cssGz, js: jsGz, json: jsonGz, total: htmlGz + cssGz + jsGz + jsonGz, js_files: js.size });
  }
  routes.sort((a, b) => a.route.localeCompare(b.route));

  const data = [];
  for (const rel of releaseDirs) {
    for (const f of walk(join(dist, 'data', rel)).filter((x) => x.endsWith('.json'))) {
      const r = relative(join(dist, 'data', rel), f).replaceAll('\\', '/');
      data.push({ path: r, gzip: gz(readFileSync(f)) });
    }
  }
  const failures = [];
  for (const r of routes) {
    if (r.total > BUDGETS.routeTotalGzip) failures.push(`${r.route}: initial transfer ${r.total} B gzip > ${BUDGETS.routeTotalGzip}`);
    if (r.js > BUDGETS.routeJsGzip) failures.push(`${r.route}: JS ${r.js} B gzip > ${BUDGETS.routeJsGzip}`);
  }
  for (const d of data) {
    const limit = d.path === 'players/index.json' ? BUDGETS.playerIndexGzip : BUDGETS.detailJsonGzip;
    if (d.gzip > limit) failures.push(`data/${d.path}: ${d.gzip} B gzip > ${limit}`);
  }
  const bytes = walk(dist).reduce((n, f) => n + statSync(f).size, 0);
  return { budgets: BUDGETS, routes, largest_data: [...data].sort((a, b) => b.gzip - a.gzip).slice(0, 10), artifact_bytes: bytes, failures };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const arg = (name, dflt) => {
    const i = process.argv.indexOf(name);
    return i >= 0 ? process.argv[i + 1] : dflt;
  };
  const dist = resolve(arg('--dir', 'dist'));
  const base = process.env.SCVIA_PUBLIC_BASE ?? '/';
  const report = measure(dist, base.endsWith('/') ? base : `${base}/`);
  const out = arg('--json', null);
  if (out) writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
  for (const r of report.routes) console.log(`${r.route.padEnd(18)} total ${(r.total / 1024).toFixed(1)} KiB  js ${(r.js / 1024).toFixed(1)} KiB  json ${(r.json / 1024).toFixed(1)} KiB`);
  console.log(`artifact ${(report.artifact_bytes / 1048576).toFixed(2)} MiB; largest data ${report.largest_data[0]?.path ?? '-'} ${((report.largest_data[0]?.gzip ?? 0) / 1024).toFixed(1)} KiB gzip`);
  if (report.failures.length) {
    console.error(`BUDGET FAILURES:\n${report.failures.join('\n')}`);
    process.exit(1);
  }
}
