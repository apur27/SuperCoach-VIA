/** Progressive enhancement for the static layout: nav disclosure, More menu, theme, zone. */
import { formatInstant, zoneLabel, type ZoneChoice } from './format';
import { getTheme, getZone, RELEASE_EVENT, setTheme, setZone, ZONE_EVENT, type ThemeChoice } from './prefs';

function rewriteTimes(zone: ZoneChoice) {
  for (const el of document.querySelectorAll<HTMLTimeElement>('time[data-instant]')) {
    const iso = el.dataset.instant;
    if (iso) {
      el.textContent = formatInstant(iso, zone);
      el.title = zoneLabel(zone);
    }
  }
}

function disclosure(button: HTMLButtonElement, panel: HTMLElement, opts: { useHidden: boolean; closeOnOutside: boolean }) {
  const setOpen = (open: boolean) => {
    button.setAttribute('aria-expanded', String(open));
    if (opts.useHidden) panel.hidden = !open;
    else panel.dataset.open = String(open);
  };
  button.addEventListener('click', () => setOpen(button.getAttribute('aria-expanded') !== 'true'));
  panel.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && button.getAttribute('aria-expanded') === 'true') {
      e.stopPropagation(); // close only the innermost open disclosure
      setOpen(false);
      button.focus();
    }
  });
  button.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setOpen(false);
  });
  if (opts.closeOnOutside) {
    document.addEventListener('click', (e) => {
      const t = e.target as Node;
      if (!panel.contains(t) && !button.contains(t)) setOpen(false);
    });
    panel.addEventListener('focusout', (e) => {
      const next = e.relatedTarget as Node | null;
      if (next && !panel.contains(next) && !button.contains(next)) setOpen(false);
    });
  }
}

export function initShell() {
  const navToggle = document.querySelector<HTMLButtonElement>('[data-nav-toggle]');
  const nav = document.getElementById('site-nav');
  if (navToggle && nav) disclosure(navToggle, nav, { useHidden: false, closeOnOutside: false });
  const moreToggle = document.querySelector<HTMLButtonElement>('[data-more-toggle]');
  const more = document.getElementById('more-list');
  if (moreToggle && more) {
    disclosure(moreToggle, more, { useHidden: true, closeOnOutside: true });
    if (more.querySelector('[aria-current="page"]')) moreToggle.textContent = 'More (current)';
  }

  const themeSel = document.querySelector<HTMLSelectElement>('[data-theme-select]');
  if (themeSel) {
    themeSel.value = getTheme();
    themeSel.addEventListener('change', () => setTheme(themeSel.value as ThemeChoice));
  }
  const zoneSel = document.querySelector<HTMLSelectElement>('[data-zone-select]');
  if (zoneSel) {
    zoneSel.value = getZone();
    zoneSel.addEventListener('change', () => setZone(zoneSel.value as ZoneChoice));
  }
  rewriteTimes(getZone());
  window.addEventListener(ZONE_EVENT, () => rewriteTimes(getZone()));

  const notice = document.querySelector<HTMLElement>('[data-release-notice]');
  window.addEventListener(RELEASE_EVENT, () => {
    if (notice && notice.hidden) {
      notice.hidden = false;
      notice.querySelector<HTMLButtonElement>('[data-reload]')?.focus();
    }
  });
  notice?.querySelector('[data-reload]')?.addEventListener('click', () => window.location.reload());
}
