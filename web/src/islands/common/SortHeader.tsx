import type { SortDir } from '../../lib/filters';

interface Props {
  label: string;
  column: string;
  sort: string | undefined;
  dir: SortDir;
  onSort: (column: string, dir: SortDir) => void;
  numeric?: boolean;
  className?: string;
}

/** Column header with aria-sort and a real button; numeric columns default to descending. */
export function SortHeader({ label, column, sort, dir, onSort, numeric = false, className }: Props) {
  const active = sort === column;
  const ariaSort = active ? (dir === 'asc' ? 'ascending' : 'descending') : 'none';
  const next: SortDir = active ? (dir === 'asc' ? 'desc' : 'asc') : numeric ? 'desc' : 'asc';
  const cls = [numeric ? 'num' : '', className ?? ''].join(' ').trim();
  return (
    <th scope="col" aria-sort={ariaSort} className={cls || undefined}>
      <button type="button" className="sort" onClick={() => onSort(column, next)}>
        {label}
        <span aria-hidden="true">{active ? (dir === 'asc' ? '▲' : '▼') : '↕'}</span>
        <span className="visually-hidden">{active ? `, sorted ${ariaSort}` : ', sortable'}</span>
      </button>
    </th>
  );
}
