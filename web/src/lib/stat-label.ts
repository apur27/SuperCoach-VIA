/** Display names stay separate from the release keys used to read values. */
export function statLabel(key: string): string {
  const words = key.replaceAll('_', ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}
