// Programmatic WCAG contrast check of the actual colour tokens in src/styles/tokens.css.
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { contrastRatio, parseThemes } from '../../src/lib/contrast';

const css = readFileSync(resolve(__dirname, '../../src/styles/tokens.css'), 'utf8');
const themes = parseThemes(css);

// [foreground, background, minimum ratio]
const PAIRS: [string, string, number][] = [
  ['--text', '--bg', 4.5], ['--text', '--surface', 4.5], ['--text-muted', '--bg', 4.5], ['--text-muted', '--surface', 4.5],
  ['--link', '--bg', 4.5], ['--link', '--surface', 4.5], ['--accent-text', '--accent', 4.5],
  ['--stale-text', '--stale-bg', 4.5], ['--error-text', '--error-bg', 4.5], ['--ok-text', '--ok-bg', 4.5],
  ['--demo-text', '--demo-bg', 4.5], ['--nav-text', '--nav-bg', 4.5], ['--header-text', '--header-bg', 4.5],
  ['--focus', '--bg', 3], ['--focus', '--header-bg', 3], ['--border-strong', '--bg', 3], ['--chart-1', '--surface', 3], ['--chart-2', '--surface', 3],
];

describe('colour tokens meet WCAG 2.2 AA', () => {
  it('parses both themes', () => {
    expect(Object.keys(themes).sort()).toEqual(['dark', 'light']);
  });
  for (const theme of ['light', 'dark'] as const) {
    for (const [fg, bg, min] of PAIRS) {
      it(`${theme}: ${fg} on ${bg} >= ${min}`, () => {
        const f = themes[theme]![fg];
        const b = themes[theme]![bg];
        expect(f, `${theme} ${fg}`).toBeTruthy();
        expect(b, `${theme} ${bg}`).toBeTruthy();
        expect(contrastRatio(f!, b!)).toBeGreaterThanOrEqual(min);
      });
    }
  }
  it('computes known ratios', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 1);
    expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.48, 1);
  });
});

describe('dark tokens are consistent', () => {
  it('system-dark block equals the explicit dark block', () => {
    const body = (re: RegExp) => {
      const m = re.exec(css)!;
      return css.slice(m.index + m[0].length, css.indexOf('}', m.index + m[0].length)).replace(/\s+/g, '');
    };
    expect(body(/:root:not\(\[data-theme="light"\]\)\s*\{/)).toBe(body(/:root\[data-theme="dark"\]\s*\{/));
  });
});
