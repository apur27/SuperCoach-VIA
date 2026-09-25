export interface NavItem { key: string; label: string; href: string }
export const PRIMARY_NAV: NavItem[] = [
  { key: 'overview', label: 'Overview', href: '' },
  { key: 'predictions', label: 'Predictions', href: 'predictions/' },
  { key: 'players', label: 'Players', href: 'players/' },
  { key: 'teams', label: 'Teams', href: 'teams/' },
  { key: 'matches', label: 'Matches', href: 'matches/' },
  { key: 'history', label: 'History', href: 'history/' },
  { key: 'accuracy', label: 'Accuracy', href: 'accuracy/' },
  { key: 'articles', label: 'Articles', href: 'articles/' },
];
export const MORE_NAV: NavItem[] = [
  { key: 'lists', label: 'Lists', href: 'lists/' },
  { key: 'watchlist', label: 'Watchlist', href: 'watchlist/' },
  { key: 'downloads', label: 'Downloads', href: 'downloads/' },
  { key: 'data-status', label: 'Data status', href: 'data-status/' },
  { key: 'methodology', label: 'Methodology', href: 'methodology/' },
];
export const THEME_KEY = 'supercoach-via:theme:v1';
export const ZONE_KEY = 'supercoach-via:zone:v1';
