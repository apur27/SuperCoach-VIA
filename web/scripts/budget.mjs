// Measure a production build against the PLAN section 12 payload budgets.
// Usage: node scripts/budget.mjs [--dir dist] [--out report.json]   (--json is an alias of --out)
// Prints a JSON report; exits 1 on any budget overrun or measurement error.
import { mkdirSync, writeFileSync } from 'node:fs';
import { cpus, totalmem } from 'node:os';
import { dirname, resolve } from 'node:path';
import { evaluate, measureSite } from './budget-lib.mjs';
import { ROUTES } from './budget-routes.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const dist = resolve(args.dir ?? process.env.SCVIA_OUT_DIR ?? 'dist');

try {
  const measured = measureSite(dist, { routes: ROUTES });
  const report = {
    ...evaluate(measured),
    dist,
    environment: { node: process.version, platform: process.platform, cpu: cpus()[0]?.model ?? null, cpus: cpus().length, ram_bytes: totalmem() },
  };
  const text = JSON.stringify(report, null, 2);
  const out = args.out ?? args.json;
  if (out) {
    mkdirSync(dirname(resolve(out)), { recursive: true });
    writeFileSync(resolve(out), `${text}\n`);
  }
  console.log(text);
  if (!report.pass) {
    console.error(`budget FAILED:\n${report.failures.join('\n')}`);
    process.exit(1);
  }
} catch (e) {
  console.error(`budget measurement failed: ${e instanceof Error ? e.message : e}`);
  process.exit(1);
}
