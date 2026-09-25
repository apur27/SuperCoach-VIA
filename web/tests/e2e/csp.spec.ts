// PLAN 11.8: meta CSP on the PRODUCTION build (both bases) with build-generated hashes,
// no 'unsafe-inline' / 'unsafe-eval', and every inline script/style covered by a hash.
import { test, expect } from './helpers';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

function htmlFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    if (statSync(p).isDirectory()) return n === 'data' || n === '_astro' ? [] : htmlFiles(p);
    return n.endsWith('.html') ? [p] : [];
  });
}
const sha = (s: string) => `'sha256-${createHash('sha256').update(s, 'utf8').digest('base64')}'`;

function parseCsp(html: string) {
  const metas = [...html.matchAll(/<meta http-equiv="content-security-policy" content="([^"]*)">/gi)];
  const directives = new Map<string, string[]>();
  for (const part of (metas[0]?.[1] ?? '').split(';').map((s) => s.trim()).filter(Boolean)) {
    const [name, ...values] = part.split(/\s+/);
    directives.set(name!, values);
  }
  return { count: metas.length, index: metas[0]?.index ?? -1, directives };
}

for (const variant of ['root', 'sub'] as const) {
  test.describe(`production CSP (${variant})`, () => {
    test.skip(({ baseURL }) => !!baseURL?.includes('/SuperCoach-VIA/'), 'file-level check of both builds; run once');
    const dist = resolve('.e2e-dist', variant);
    const files = htmlFiles(dist);

    test('every page has one meta CSP with the required directives and no unsafe sources', () => {
      expect(files.length).toBeGreaterThan(15);
      for (const f of files) {
        const html = readFileSync(f, 'utf8');
        const rel = relative(dist, f);
        const { count, index, directives: d } = parseCsp(html);
        expect(count, rel).toBe(1);
        // Everything that executes must come after the policy, or the policy does not govern it.
        const firstScript = html.search(/<script\b/);
        const firstStyle = html.search(/<style\b|<link rel="stylesheet"/);
        if (firstScript >= 0) expect(index, `${rel}: CSP after first script`).toBeLessThan(firstScript);
        if (firstStyle >= 0) expect(index, `${rel}: CSP after first style`).toBeLessThan(firstStyle);
        expect(d.get('default-src'), rel).toEqual(["'self'"]);
        expect(d.get('object-src'), rel).toEqual(["'none'"]);
        expect(d.get('base-uri'), rel).toEqual(["'self'"]);
        expect(d.get('connect-src'), rel).toEqual(["'self'"]);
        for (const name of ['script-src', 'style-src']) {
          const v = d.get(name) ?? [];
          expect(v[0], `${rel} ${name}`).toBe("'self'");
          for (const src of v.slice(1)) expect(src, `${rel} ${name}`).toMatch(/^'sha256-[A-Za-z0-9+/=]+'$/);
        }
        for (const bad of ["'unsafe-inline'", "'unsafe-eval'", "'unsafe-hashes'", "'strict-dynamic'", '*', 'data:', 'http:', 'https:']) {
          expect(d.get('script-src') ?? [], `${rel} script-src ${bad}`).not.toContain(bad);
        }
        expect(d.get('style-src') ?? [], rel).not.toContain("'unsafe-inline'");
      }
    });

    test('every inline script and style is hash-listed; no inline handlers, style attributes or external origins', () => {
      for (const f of files) {
        const html = readFileSync(f, 'utf8');
        const rel = relative(dist, f);
        const { directives: d } = parseCsp(html);
        const scriptSrc = d.get('script-src') ?? [];
        const styleSrc = d.get('style-src') ?? [];
        for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
          if (/\bsrc=/.test(m[1]!)) continue;
          expect(scriptSrc, `${rel}: inline script not hashed`).toContain(sha(m[2]!));
        }
        for (const m of html.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)) {
          expect(styleSrc, `${rel}: inline style not hashed`).toContain(sha(m[1]!));
        }
        expect(html, `${rel}: inline event handler`).not.toMatch(/<[a-z][^>]*\son[a-z]+\s*=/i);
        expect(html, `${rel}: style attribute`).not.toMatch(/<[a-z][^>]*\sstyle="/i);
        expect(html, `${rel}: javascript: URL`).not.toMatch(/(href|src)="\s*javascript:/i);
        for (const m of html.matchAll(/<(?:script|link|img|iframe)\b[^>]*\s(?:src|href)="([^"]+)"/g)) {
          expect(m[1]!, `${rel}: external resource`).not.toMatch(/^(https?:)?\/\//);
        }
      }
    });
  });
}

test('the browser enforces the policy: an injected inline script is refused', async ({ page }) => {
  await page.goto('');
  const result = await page.evaluate(async () => {
    const w = window as unknown as { __injected?: boolean };
    const violation = new Promise<string>((res) => document.addEventListener('securitypolicyviolation', (e) => res(e.violatedDirective), { once: true }));
    const s = document.createElement('script');
    s.textContent = 'window.__injected = true';
    document.body.append(s);
    const directive = await Promise.race([violation, new Promise<string>((r) => setTimeout(() => r('none'), 2000))]);
    return { ran: w.__injected === true, directive };
  });
  expect(result.ran).toBe(false);
  expect(result.directive).toMatch(/script-src/);
});
