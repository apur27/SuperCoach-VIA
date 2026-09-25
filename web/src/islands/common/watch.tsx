import { useEffect, useState } from 'react';
import { browserWatchlist } from '../../lib/watchlist';

export function useWatchlist() {
  const [ids, setIds] = useState<string[]>([]);
  const [persistent, setPersistent] = useState(true);
  useEffect(() => {
    const w = browserWatchlist();
    setIds(w.list());
    setPersistent(w.persistent());
    return w.subscribe((next) => {
      setIds(next);
      setPersistent(w.persistent());
    });
  }, []);
  return { ids, persistent, store: typeof window === 'undefined' ? null : browserWatchlist() };
}

export function WatchButton({ id, name, compact = false }: { id: string; name: string; compact?: boolean }) {
  const { ids, store } = useWatchlist();
  const on = ids.includes(id);
  const full = ids.length >= 100 && !on;
  return (
    <button
      type="button"
      className={compact ? 'secondary small' : 'secondary'}
      aria-pressed={on}
      aria-label={compact ? `Watch ${name}` : undefined}
      disabled={full}
      title={full ? 'Watchlist is full (100 players)' : undefined}
      onClick={() => (on ? store?.remove(id) : store?.add(id))}
    >
      {compact ? (on ? '★ Watching' : '☆ Watch') : on ? '★ On watchlist (remove)' : '☆ Add to watchlist'}
    </button>
  );
}
