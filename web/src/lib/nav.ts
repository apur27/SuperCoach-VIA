export interface NavItem { key: string; label: string; href: string }
export const PRIMARY_NAV: NavItem[] = [
  { key: 'overview', label: 'Home', href: '' },
  { key: 'players', label: 'Players', href: 'players/' },
  { key: 'matches', label: 'Matches', href: 'matches/' },
  { key: 'history', label: 'Rankings', href: 'history/' },
  { key: 'predictions', label: 'Predictions', href: 'predictions/' },
];
export const MORE_NAV: NavItem[] = [
  { key: 'teams', label: 'Teams', href: 'teams/' },
  { key: 'accuracy', label: 'Accuracy', href: 'accuracy/' },
  { key: 'articles', label: 'Articles', href: 'articles/' },
  { key: 'lists', label: 'Lists', href: 'lists/' },
  { key: 'watchlist', label: 'Watchlist', href: 'watchlist/' },
  { key: 'downloads', label: 'Downloads', href: 'downloads/' },
  { key: 'data-status', label: 'Data status', href: 'data-status/' },
  { key: 'methodology', label: 'Methodology', href: 'methodology/' },
];
export const THEME_KEY = 'supercoach-via:theme:v1';
export const ZONE_KEY = 'supercoach-via:zone:v1';
