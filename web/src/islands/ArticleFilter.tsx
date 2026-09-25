import type { ArticleSummary } from '../lib/contracts';
import { parseUrlState, serializeUrlState, type StateSpec } from '../lib/filters';
import { withBase } from '../lib/ids';
import { formatDateOnly } from '../lib/format';
import { normalizeText } from '../lib/search';
import { useUrlSearch } from './common/runtime';

const SCOPE = { frozen: 'Frozen as of publication', live: 'Live (regenerated each release)', archive: 'Archive' } as const;

/** Server-rendered list (readable without JS); hydrated for search/category/year filters. */
export default function ArticleFilter({ articles, base }: { articles: ArticleSummary[]; base: string }) {
  const [search, setSearch] = useUrlSearch();
  const spec = { q: { kind: 'text', maxLength: 60 }, category: { kind: 'token' }, year: { kind: 'int', min: 1850, max: 2100 } } as const satisfies StateSpec;
  const state = parseUrlState(search, spec);
  const categories = [...new Set(articles.map((a) => a.category))].sort();
  const years = [...new Set(articles.map((a) => a.published?.slice(0, 4)).filter((y): y is string => Boolean(y)))].sort().reverse();
  const q = normalizeText(state.q ?? '');
  const shown = articles.filter((a) => (!q || normalizeText(`${a.title} ${a.excerpt}`).includes(q)) && (!state.category || a.category === state.category) && (!state.year || a.published?.startsWith(String(state.year))));
  const update = (patch: Partial<typeof state>, mode: 'push' | 'replace' = 'push') => setSearch(serializeUrlState({ ...state, ...patch }, spec), mode);
  return (
    <div>
      <form className="controls" role="search" onSubmit={(e) => e.preventDefault()}>
        <div className="field"><label htmlFor="a-q">Search articles</label><input id="a-q" name="q" type="search" defaultValue={state.q ?? ''} key={state.q ?? ''} maxLength={60} onChange={(e) => update({ q: e.target.value.trim() || undefined }, 'replace')} /></div>
        <div className="field"><label htmlFor="a-cat">Category</label>
          <select id="a-cat" value={state.category ?? ''} onChange={(e) => update({ category: e.target.value || undefined })}><option value="">All categories</option>{categories.map((c) => <option key={c} value={c}>{c}</option>)}</select></div>
        <div className="field"><label htmlFor="a-year">Year</label>
          <select id="a-year" value={state.year ?? ''} onChange={(e) => update({ year: e.target.value ? Number(e.target.value) : undefined })}><option value="">All years</option>{years.map((y) => <option key={y} value={y}>{y}</option>)}</select></div>
      </form>
      <p role="status" aria-live="polite">{shown.length} of {articles.length} articles</p>
      {shown.length === 0 ? <p data-state="empty">No articles match.</p> : (
        <ul className="list-plain">
          {shown.map((a) => (
            <li key={a.slug}>
              <h2><a href={withBase(base, `articles/${a.slug}/`)}>{a.title}</a></h2>
              <p className="muted">{a.category} · published {formatDateOnly(a.published)} · {SCOPE[a.scope]}{a.as_of ? ` · ${a.as_of}` : ''}{a.editorial_state === 'draft' ? ' · DRAFT' : ''}</p>
              <p>{a.excerpt}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
