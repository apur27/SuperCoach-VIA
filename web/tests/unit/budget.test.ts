import { describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomBytes } from 'node:crypto';
import { BUDGETS, measure } from '../../scripts/budget.mjs';

function site(): string {
  const d = mkdtempSync(join(tmpdir(), 'budget-'));
  mkdirSync(join(d, '_astro'));
  mkdirSync(join(d, 'data/r1/players'), { recursive: true });
  writeFileSync(join(d, 'index.html'), '<link rel="stylesheet" href="/_astro/a.css"><astro-island component-url="/_astro/entry.js"></astro-island>');
  writeFileSync(join(d, '_astro/a.css'), 'body{}');
  writeFileSync(join(d, '_astro/entry.js'), 'import{x}from"./dep.js";import("./lazy.js");');
  writeFileSync(join(d, '_astro/dep.js'), 'export const x=1;');
  writeFileSync(join(d, '_astro/lazy.js'), randomBytes(200_000).toString('hex'));
  writeFileSync(join(d, 'data/r1/overview.json'), '{}');
  return d;
}

describe('payload budget', () => {
  it('counts static import closure and initial JSON but not lazy chunks', () => {
    const r = measure(site(), '/');
    const home = r.routes.find((x: { route: string }) => x.route === '/');
    expect(home?.js_files).toBe(2);
    expect(home?.json).toBeGreaterThan(0);
    expect(r.failures).toEqual([]);
  });

  it('fails an oversized detail JSON', () => {
    const d = site();
    writeFileSync(join(d, 'data/r1/players/p.json'), JSON.stringify(randomBytes(BUDGETS.detailJsonGzip).toString('base64')));
    expect(measure(d, '/').failures.some((f: string) => f.includes('players/p.json'))).toBe(true);
  });
});
