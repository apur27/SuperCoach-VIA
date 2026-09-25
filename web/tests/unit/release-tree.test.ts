// integrations/release-tree.mjs: the site consumes an external (Python-built) release via
// SCVIA_RELEASE_DIR, verifies it against its manifest and copies it under data/<release_id>/.
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import {
  DEFAULT_RELEASE_DIR, copyRelease, isSafeRelPath, readManifest, releaseDirFromEnv, retainedDirsFromEnv, verifyManifest, walkRelease,
} from '../../integrations/release-tree.mjs';

const WEB = resolve(__dirname, '../..');
let tmp: string;
beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), 'scvia-release-'));
});
afterEach(() => rmSync(tmp, { recursive: true, force: true }));

// vitest exports Vite env (BASE_URL, MODE, NODE_ENV=test, ...) into process.env; a real build must not inherit it.
const VITEST_ENV = /^(BASE_URL|MODE|DEV|PROD|SSR|NODE_ENV|VITEST.*|TEST)$/;
const buildEnv = (extra: Record<string, string>) => ({
  ...Object.fromEntries(Object.entries(process.env).filter(([k]) => !VITEST_ENV.test(k))),
  ...extra,
});
const sha = (b: Buffer) => createHash('sha256').update(b).digest('hex');

function writeRelease(dir: string, id: string, files: Record<string, string>) {
  mkdirSync(dir, { recursive: true });
  const resources: Record<string, { path: string; sha256: string; bytes: number }> = {};
  for (const [rel, body] of Object.entries(files)) {
    mkdirSync(join(dir, rel, '..'), { recursive: true });
    writeFileSync(join(dir, rel), body);
    resources[rel.replace(/\W+/g, '_')] = { path: rel, sha256: sha(Buffer.from(body)), bytes: Buffer.byteLength(body) };
  }
  writeFileSync(join(dir, 'release.json'), JSON.stringify({ release_id: id, resources }));
}

/** Copy the DEMO fixture as if it were a separately built release with a new ID. */
function externalRelease(dir: string, id: string) {
  cpSync(DEFAULT_RELEASE_DIR, dir, { recursive: true });
  const old = readManifest(dir).release_id as string;
  const manifest = JSON.parse(readFileSync(join(dir, 'release.json'), 'utf8'));
  manifest.release_id = id;
  for (const f of ['overview.json', 'downloads.json']) {
    const body = Buffer.from(readFileSync(join(dir, f), 'utf8').replaceAll(old, id));
    writeFileSync(join(dir, f), body);
    for (const ref of Object.values(manifest.resources) as { path: string; sha256: string; bytes: number }[]) {
      if (ref.path === f) Object.assign(ref, { sha256: sha(body), bytes: body.length });
    }
  }
  writeFileSync(join(dir, 'release.json'), JSON.stringify(manifest));
}

describe('environment', () => {
  it('defaults to the DEMO fixture and requires absolute paths', () => {
    expect(releaseDirFromEnv({})).toBe(DEFAULT_RELEASE_DIR);
    expect(releaseDirFromEnv({ SCVIA_RELEASE_DIR: '/abs/release' })).toBe('/abs/release');
    expect(() => releaseDirFromEnv({ SCVIA_RELEASE_DIR: 'relative/release' })).toThrow(/absolute/);
    expect(retainedDirsFromEnv({ SCVIA_RETAINED_RELEASE_DIRS: '/a:/b' })).toEqual(['/a', '/b']);
    expect(() => retainedDirsFromEnv({ SCVIA_RETAINED_RELEASE_DIRS: '/a:b' })).toThrow(/absolute/);
  });
  it('rejects traversal and odd characters in resource paths', () => {
    for (const bad of ['../x.json', '/x.json', 'a/../b.json', 'a\\b.json', 'https://x/y.json', 'a b.json', 'a%2e.json', '.hidden.json']) {
      expect(isSafeRelPath(bad), bad).toBe(false);
    }
    expect(isSafeRelPath('players/legacy__a.json')).toBe(true);
  });
});

describe('manifest verification and copy', () => {
  it('detects missing, resized and altered resources', () => {
    const dir = join(tmp, 'r');
    writeRelease(dir, 'r1', { 'a.json': '{"a":1}', 'b/c.json': '{"c":2}' });
    expect(verifyManifest(dir, readManifest(dir))).toEqual([]);
    writeFileSync(join(dir, 'a.json'), '{"a":2}');
    expect(verifyManifest(dir, readManifest(dir)).join('\n')).toMatch(/sha256 mismatch/);
    writeFileSync(join(dir, 'a.json'), '{"a":10}');
    expect(verifyManifest(dir, readManifest(dir)).join('\n')).toMatch(/size/);
    rmSync(join(dir, 'b/c.json'));
    expect(verifyManifest(dir, readManifest(dir)).join('\n')).toMatch(/missing b\/c.json/);
  });
  it('refuses an unsafe release_id', () => {
    const dir = join(tmp, 'r');
    writeRelease(dir, '../escape', {});
    expect(() => readManifest(dir)).toThrow(/unsafe release_id/);
  });
  it('refuses symlinks and unexpected file types in the release tree', () => {
    const dir = join(tmp, 'r');
    writeRelease(dir, 'r1', { 'a.json': '{}' });
    symlinkSync('/etc/passwd', join(dir, 'link.json'));
    expect(() => walkRelease(dir)).toThrow(/symlink/);
    rmSync(join(dir, 'link.json'));
    writeFileSync(join(dir, 'model.pkl'), 'x');
    expect(() => walkRelease(dir)).toThrow(/unexpected file type/);
  });
  it('copies a verified release under data/<release_id>/ and refuses a tampered one', () => {
    const dir = join(tmp, 'r');
    writeRelease(dir, 'r1', { 'a.json': '{"a":1}' });
    const site = join(tmp, 'site');
    expect(copyRelease(dir, site)).toEqual({ releaseId: 'r1', files: 2 });
    expect(readFileSync(join(site, 'data/r1/a.json'), 'utf8')).toBe('{"a":1}');
    writeFileSync(join(dir, 'a.json'), '{"a":9}');
    expect(() => copyRelease(dir, join(tmp, 'site2'))).toThrow(/failed verification/);
  });
});

describe('production build from an external release (SCVIA_RELEASE_DIR)', () => {
  it('embeds the external release ID, serves its data and keeps retained releases alongside', () => {
    const ext = join(tmp, 'external');
    externalRelease(ext, '20990101T000000Z-external');
    const out = join(tmp, 'dist');
    const r = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], {
      cwd: WEB, encoding: 'utf8', timeout: 120000,
      env: buildEnv({ SCVIA_RELEASE_DIR: ext, SCVIA_RETAINED_RELEASE_DIRS: DEFAULT_RELEASE_DIR, SCVIA_OUT_DIR: out, SCVIA_PUBLIC_BASE: '/SuperCoach-VIA/' }),
    });
    expect(r.status, r.stderr || r.stdout).toBe(0);
    const html = readFileSync(join(out, 'index.html'), 'utf8');
    expect(html).toContain('data-release="20990101T000000Z-external"');
    expect(html).toContain('data-base="/SuperCoach-VIA/"');
    expect(existsSync(join(out, 'data/20990101T000000Z-external/release.json'))).toBe(true);
    expect(existsSync(join(out, `data/${readManifest(DEFAULT_RELEASE_DIR).release_id}/release.json`))).toBe(true);
  }, 120000);

  it('fails the build when the external release does not verify', () => {
    const ext = join(tmp, 'external');
    externalRelease(ext, '20990101T000000Z-bad');
    writeFileSync(join(ext, 'overview.json'), '{}');
    const r = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], {
      cwd: WEB, encoding: 'utf8', timeout: 120000, env: buildEnv({ SCVIA_RELEASE_DIR: ext, SCVIA_OUT_DIR: join(tmp, 'dist') }),
    });
    expect(r.status).not.toBe(0);
    expect(`${r.stdout}${r.stderr}`).toMatch(/release manifest verification failed/);
  }, 120000);
});
