import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { assertSanitizedHtml, findUnsafeHtml } from '../../src/lib/html-allowlist';

describe('article html allowlist (defence in depth; Python sanitizes at build)', () => {
  it('accepts allowlisted markup', () => {
    expect(findUnsafeHtml('<h2>x</h2><p>a <a href="../y/">b</a> <a href="https://example.org/x">c</a></p><ul><li>i</li></ul><table><tr><th scope="col">h</th></tr></table>')).toEqual([]);
  });
  it.each([
    ['<script>alert(1)</script>', 'tag script'],
    ['<img src=x onerror=alert(1)>', 'tag img'],
    ['<p onclick="x()">a</p>', 'attribute onclick'],
    ['<a href="javascript:alert(1)">x</a>', 'url javascript:alert(1)'],
    ['<a href=" JaVaScRiPt:alert(1)">x</a>', 'url'],
    ['<a href="data:text/html,x">x</a>', 'url data:text/html,x'],
    ['<iframe src="//x"></iframe>', 'tag iframe'],
    ['<style>p{}</style>', 'tag style'],
    ['<p style="color:red">x</p>', 'attribute style'],
    ['<svg><use href="#x"/></svg>', 'tag svg'],
    ['<!-- <script> -->', 'comment'],
    ['<p>x</p><form action="/"></form>', 'tag form'],
  ])('rejects %s', (html, reason) => {
    const issues = findUnsafeHtml(html);
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.join(' ')).toContain(reason);
    expect(() => assertSanitizedHtml(html, 't')).toThrow();
  });
  it('every fixture article passes', () => {
    const dir = resolve(__dirname, '../fixtures/demo-release/articles');
    for (const f of readdirSync(dir).filter((x) => x !== 'index.json')) {
      const art = JSON.parse(readFileSync(resolve(dir, f), 'utf8'));
      expect(findUnsafeHtml(art.html), f).toEqual([]);
    }
  });
});
