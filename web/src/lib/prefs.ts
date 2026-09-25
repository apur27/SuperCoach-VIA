/** Per-viewer preferences (theme, time zone). Storage failures fall back silently. */
import type { ZoneChoice } from './format';
import { DEFAULT_ZONE } from './format';
import { THEME_KEY, ZONE_KEY } from './nav';

export type ThemeChoice = 'system' | 'light' | 'dark';
export const ZONE_EVENT = 'scvia:zone';
export const RELEASE_EVENT = 'scvia:release-unavailable';

function get(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}
function set(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage disabled: preference lasts for this page only */
  }
}
let memoryZone: ZoneChoice | null = null;

export function getZone(): ZoneChoice {
  const v = memoryZone ?? get(ZONE_KEY);
  return v === 'local' || v === 'UTC' || v === 'Australia/Melbourne' ? v : DEFAULT_ZONE;
}
export function setZone(z: ZoneChoice) {
  memoryZone = z;
  set(ZONE_KEY, z);
  window.dispatchEvent(new CustomEvent(ZONE_EVENT, { detail: z }));
}
export function getTheme(): ThemeChoice {
  const v = get(THEME_KEY);
  return v === 'light' || v === 'dark' ? v : 'system';
}
export function setTheme(t: ThemeChoice) {
  if (t === 'system') {
    document.documentElement.removeAttribute('data-theme');
    try {
      window.localStorage.removeItem(THEME_KEY);
    } catch {
      /* ignore */
    }
  } else {
    document.documentElement.setAttribute('data-theme', t);
    set(THEME_KEY, t);
  }
}
/** Ask the shell to show the "newer release available" notice. */
export function announceReleaseUnavailable() {
  window.dispatchEvent(new CustomEvent(RELEASE_EVENT));
}
