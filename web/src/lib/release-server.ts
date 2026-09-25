/** Build-time (Node) access to the input release. Never bundled for the browser. */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ReleaseManifest, ResourceKind, ResourceTypes } from './contracts';
import { validate } from './contracts';
import { isSafeResourcePath } from './ids';
// @ts-expect-error untyped build helper shared with astro.config.mjs
import { retainedDirsFromEnv, readManifest } from '../../integrations/release-tree.mjs';

function resolvedDir(): string {
  const d = process.env.SCVIA_RELEASE_DIR_RESOLVED;
  if (!d) throw new Error('SCVIA_RELEASE_DIR_RESOLVED is not set (it is set by astro.config.mjs)');
  return d;
}
const dir: string = resolvedDir();
const cache = new Map<string, unknown>();

export function releaseDir(): string {
  return dir;
}

export function readResource<K extends ResourceKind>(kind: K, path: string): ResourceTypes[K] {
  if (!isSafeResourcePath(path)) throw new Error(`unsafe resource path: ${path}`);
  const key = `${kind}:${path}`;
  if (cache.has(key)) return cache.get(key) as ResourceTypes[K];
  const data: unknown = JSON.parse(readFileSync(join(dir, path), 'utf8'));
  const res = validate(kind, data);
  if (!res.ok) throw new Error(`release resource ${path} failed ${kind} validation: ${res.error}`);
  cache.set(key, res.value);
  return res.value;
}

export function hasResource(path: string): boolean {
  return isSafeResourcePath(path) && existsSync(join(dir, path));
}

export function manifest(): ReleaseManifest {
  return readResource('release', 'release.json');
}

export function overview() {
  return readResource('overview', 'overview.json');
}

/** Season list from `match_index:<season>` manifest keys, newest first. */
export function matchSeasons(): number[] {
  return Object.keys(manifest().resources)
    .map((k) => /^match_index:(\d{4})$/.exec(k)?.[1])
    .filter((s): s is string => Boolean(s))
    .map(Number)
    .sort((a, b) => b - a);
}

export interface RetainedInfo { releaseId: string; generatedAt: string }
export function retainedReleases(): RetainedInfo[] {
  return (retainedDirsFromEnv() as string[]).map((d) => {
    const m = readManifest(d) as { release_id: string; generated_at: string };
    return { releaseId: m.release_id, generatedAt: m.generated_at };
  });
}
