interface Props {
  page: number;
  pages: number;
  total: number;
  size: string;
  sizes?: readonly string[];
  onPage: (p: number) => void;
  onSize: (s: string) => void;
  label: string;
}
export function Pagination({ page, pages, total, size, sizes = ['25', '50', '100'], onPage, onSize, label }: Props) {
  return (
    <nav className="pagination" aria-label={`${label} pages`}>
      <button type="button" className="secondary small" disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</button>
      <span className="tabular">Page {page} of {pages} ({total} total)</span>
      <button type="button" className="secondary small" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</button>
      <label>
        Rows per page{' '}
        <select value={size} onChange={(e) => onSize(e.target.value)}>
          {sizes.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </label>
    </nav>
  );
}
