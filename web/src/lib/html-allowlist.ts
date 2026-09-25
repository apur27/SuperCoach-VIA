/**
 * Defensive allowlist check for article HTML. The Python side sanitizes at build time;
 * this refuses to render anything outside the allowlist so a sanitizer regression fails
 * the site build instead of shipping active content.
 */
const ALLOWED_TAGS = new Set([
  'a', 'abbr', 'b', 'blockquote', 'br', 'caption', 'code', 'dd', 'del', 'div', 'dl', 'dt', 'em', 'figcaption', 'figure',
  'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'li', 'ol', 'p', 'pre', 's', 'small', 'span', 'strong', 'sub', 'sup',
  'table', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'ul',
]);
const ALLOWED_ATTRS: Record<string, Set<string>> = {
  '*': new Set(['title', 'lang', 'dir']),
  a: new Set(['href', 'title', 'rel']),
  th: new Set(['scope', 'colspan', 'rowspan', 'abbr']),
  td: new Set(['colspan', 'rowspan']),
  ol: new Set(['start', 'reversed']),
};
const SAFE_URL_RE = /^(https?:\/\/|mailto:|#|\.{0,2}\/|[A-Za-z0-9_-][A-Za-z0-9_\-./]*(#[A-Za-z0-9_-]*)?$)/;

export function findUnsafeHtml(html: string): string[] {
  const issues: string[] = [];
  if (/<!--/.test(html)) issues.push('comment');
  if (/<!\[CDATA\[|<\?|<!doctype/i.test(html)) issues.push('declaration');
  const tagRe = /<\/?([A-Za-z][A-Za-z0-9-]*)([^>]*)>/g;
  let m: RegExpExecArray | null;
  while ((m = tagRe.exec(html))) {
    const tag = m[1]!.toLowerCase();
    if (!ALLOWED_TAGS.has(tag)) {
      issues.push(`tag ${tag}`);
      continue;
    }
    const attrText = m[2] ?? '';
    const attrRe = /([^\s"'=<>/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;
    let a: RegExpExecArray | null;
    while ((a = attrRe.exec(attrText))) {
      const name = a[1]!.toLowerCase();
      const value = a[2] ?? a[3] ?? a[4] ?? '';
      const allowed = ALLOWED_ATTRS[tag]?.has(name) || ALLOWED_ATTRS['*']!.has(name);
      if (!allowed) {
        issues.push(`attribute ${name} on ${tag}`);
        continue;
      }
      if (name === 'href') {
        const decoded = value.replace(/&#x?[0-9a-f]+;?|&[a-z]+;/gi, '?').trim();
        if (decoded !== value.trim() || !SAFE_URL_RE.test(value.trim()) || /^[a-z][a-z0-9+.-]*:/i.test(value.trim()) && !/^(https?|mailto):/i.test(value.trim())) {
          issues.push(`url ${value.trim()}`);
        }
      }
    }
  }
  // A '<' that does not start a recognised tag could hide markup from the scanner.
  if (/<(?![A-Za-z/])/.test(html.replace(/&lt;/g, ''))) issues.push('stray angle bracket');
  return issues;
}

export function assertSanitizedHtml(html: string, where: string): string {
  const issues = findUnsafeHtml(html);
  if (issues.length) throw new Error(`unsanitized article HTML in ${where}: ${issues.slice(0, 5).join('; ')}`);
  return html;
}
