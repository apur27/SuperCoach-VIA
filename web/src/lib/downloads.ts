import type { Downloads } from './contracts';
/** Find a download item by key (e.g. "predictions_csv"), else the first of a kind. */
export function findDownload(d: Downloads, key: string, kind?: string) {
  return d.items.find((i) => i.key === key) ?? (kind ? d.items.find((i) => i.kind === kind) : undefined);
}
