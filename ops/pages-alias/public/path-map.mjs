/** Return a same-origin project path only for a differently cased first segment. */
export function aliasTarget(pathname, search = '', hash = '') {
  if ((search && !search.startsWith('?')) || (hash && !hash.startsWith('#'))) return null;
  const match = /^\/(supercoach-via)(\/.*)?$/i.exec(pathname);
  if (!match || match[1] === 'SuperCoach-VIA') return null;
  const targetPath = `/SuperCoach-VIA${match[2] ?? '/'}`;
  // The browser normalizes dot segments and backslashes. Refuse a path whose
  // interpretation changes so it cannot escape the fixed project prefix.
  if (new URL(targetPath, 'https://alias.invalid').pathname !== targetPath) return null;
  return `${targetPath}${search}${hash}`;
}
