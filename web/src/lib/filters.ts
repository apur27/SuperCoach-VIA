/** URL state, sorting and pagination shared by all explorers (PLAN 9.3). */

export type FieldSpec =
  | { kind: 'int'; min: number; max: number; default?: number }
  | { kind: 'token'; default?: string }
  | { kind: 'text'; maxLength: number; default?: string }
  | { kind: 'enum'; values: readonly string[]; default?: string }
  | { kind: 'ids'; max: number };

export type StateSpec = Record<string, FieldSpec>;

type FieldValue<F> = F extends { kind: 'int' }
  ? number
  : F extends { kind: 'enum'; values: readonly (infer V)[] }
    ? V
    : F extends { kind: 'ids' }
      ? string[]
      : string;

export type UrlState<S extends StateSpec> = { [K in keyof S]?: FieldValue<S[K]> | undefined };

const TOKEN_RE = /^[A-Za-z0-9][A-Za-z0-9_\-.:]{0,159}$/;

function parseField(spec: FieldSpec, raw: string | null): unknown {
  if (raw === null) return spec.kind === 'ids' ? undefined : spec.default;
  switch (spec.kind) {
    case 'int': {
      if (!/^-?\d{1,9}$/.test(raw)) return spec.default;
      const n = Number(raw);
      return n >= spec.min && n <= spec.max ? n : spec.default;
    }
    case 'token':
      return TOKEN_RE.test(raw) && !raw.includes('..') ? raw : spec.default;
    case 'text': {
      // Stripping control characters from URL text is the point of this pattern.
      // eslint-disable-next-line no-control-regex
      const t = raw.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, spec.maxLength);
      return t ? t : spec.default;
    }
    case 'enum':
      return spec.values.includes(raw) ? raw : spec.default;
    case 'ids': {
      const ids = raw.split(',').map((s) => s.trim()).filter(Boolean);
      const valid = ids.filter((id) => TOKEN_RE.test(id) && !id.includes('..'));
      return [...new Set(valid)].slice(0, spec.max);
    }
  }
}

export function parseUrlState<S extends StateSpec>(search: string, spec: S): UrlState<S> {
  const params = new URLSearchParams(search);
  const out: Record<string, unknown> = {};
  for (const [name, field] of Object.entries(spec)) {
    const v = parseField(field, params.get(name));
    if (v !== undefined) out[name] = v;
  }
  return out as UrlState<S>;
}

export function serializeUrlState<S extends StateSpec>(state: UrlState<S>, spec: S): string {
  const params = new URLSearchParams();
  for (const [name, field] of Object.entries(spec)) {
    const v = (state as Record<string, unknown>)[name];
    if (v === undefined || v === null || v === '') continue;
    if (Array.isArray(v)) {
      if (v.length) params.set(name, v.join(','));
      continue;
    }
    if ('default' in field && field.default !== undefined && v === field.default) continue;
    params.set(name, String(v));
  }
  const s = params.toString();
  return s ? `?${s}` : '';
}

export type SortDir = 'asc' | 'desc';
type Sortable = number | string | null | undefined;

const collator = new Intl.Collator('en-AU', { sensitivity: 'base', numeric: true });

export function compareValues(a: Sortable, b: Sortable): number {
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return collator.compare(String(a), String(b));
}

/** Stable sort; null/undefined/NaN always last regardless of direction. */
export function sortRows<T>(rows: readonly T[], key: (row: T) => Sortable, dir: SortDir): T[] {
  const isMissing = (v: Sortable) => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
  return rows
    .map((row, i) => ({ row, i, v: key(row) }))
    .sort((x, y) => {
      const mx = isMissing(x.v);
      const my = isMissing(y.v);
      if (mx || my) return mx === my ? x.i - y.i : mx ? 1 : -1;
      const c = compareValues(x.v, y.v);
      return c === 0 ? x.i - y.i : dir === 'asc' ? c : -c;
    })
    .map((x) => x.row);
}

export function paginate<T>(items: readonly T[], page: number, size: number) {
  const pages = Math.max(1, Math.ceil(items.length / size));
  const p = Math.min(Math.max(1, page), pages);
  return { page: p, pages, total: items.length, items: items.slice((p - 1) * size, p * size) };
}
