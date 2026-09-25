/** WCAG relative-luminance contrast (used by the token test). */
function channel(c: number): number {
  const s = c / 255;
  return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
}
function luminance(hex: string): number {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) throw new Error(`expected #rrggbb, got ${hex}`);
  const n = parseInt(m[1]!, 16);
  return 0.2126 * channel((n >> 16) & 255) + 0.7152 * channel((n >> 8) & 255) + 0.0722 * channel(n & 255);
}
export function contrastRatio(a: string, b: string): number {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p) as [number, number];
  return (x + 0.05) / (y + 0.05);
}
/** Extract `--token: #hex` declarations from the light (:root) and dark ([data-theme="dark"]) blocks. */
export function parseThemes(css: string): Record<'light' | 'dark', Record<string, string>> {
  const block = (selector: RegExp) => {
    const m = selector.exec(css);
    if (!m) return {};
    const body = css.slice(m.index + m[0].length, css.indexOf('}', m.index));
    return Object.fromEntries([...body.matchAll(/(--[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;/g)].map((x) => [x[1]!, x[2]!]));
  };
  return { light: block(/:root\s*\{/), dark: block(/:root\[data-theme="dark"\]\s*\{/) };
}
