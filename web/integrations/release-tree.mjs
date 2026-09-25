// Build-time release handling shared by astro.config.mjs and scripts.
// - Resolves the input release (SCVIA_RELEASE_DIR or the DEMO fixture).
// - Verifies the manifest: safe relative paths, sizes and sha256 of listed resources.
// - Copies the release tree (and retained releases) into <site>/data/<release_id>/,
//   refusing symlinks, traversal and unexpected file types.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_RELEASE_DIR = resolve(here, '../tests/fixtures/demo-release');
const SAFE_KEY_RE = /^[A-Za-z0-9][A-Za-z0-9_\-.]{0,159}$/;
const SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9_\-.]*$/;
const ALLOWED_EXT = new Set(['.json', '.csv', '.zip', '.md', '.png', '.svg', '.txt']);

export function releaseDirFromEnv(env = process.env) {
  const dir = env.SCVIA_RELEASE_DIR;
  if (!dir) return DEFAULT_RELEASE_DIR;
  if (!isAbsolute(dir)) throw new Error('SCVIA_RELEASE_DIR must be an absolute path');
  return dir;
}

export function retainedDirsFromEnv(env = process.env) {
  return (env.SCVIA_RETAINED_RELEASE_DIRS ?? '').split(':').filter(Boolean).map((d) => {
    if (!isAbsolute(d)) throw new Error('SCVIA_RETAINED_RELEASE_DIRS entries must be absolute paths');
    return d;
  });
}

export function isSafeRelPath(p) {
  return typeof p === 'string' && p.length > 0 && p.length <= 512 && !p.startsWith('/') && !p.includes('\\') && !p.includes('://') &&
    !/[?#%\s]/.test(p) && p.split('/').every((s) => SEGMENT_RE.test(s) && !s.includes('..'));
}

export function readManifest(dir) {
  const path = join(dir, 'release.json');
  if (!existsSync(path)) throw new Error(`no release.json in ${dir}`);
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  if (!SAFE_KEY_RE.test(manifest.release_id ?? '') || manifest.release_id.includes('..')) {
    throw new Error(`unsafe release_id in ${path}`);
  }
  return manifest;
}

/** Verify listed resources exist with the declared size and sha256. Returns problems. */
export function verifyManifest(dir, manifest) {
  const problems = [];
  for (const [key, ref] of Object.entries(manifest.resources ?? {})) {
    if (!isSafeRelPath(ref.path)) {
      problems.push(`${key}: unsafe path ${ref.path}`);
      continue;
    }
    const file = join(dir, ref.path);
    if (!existsSync(file) || !lstatSync(file).isFile()) {
      problems.push(`${key}: missing ${ref.path}`);
      continue;
    }
    const bytes = readFileSync(file);
    if (bytes.length !== ref.bytes) problems.push(`${key}: size ${bytes.length} != ${ref.bytes}`);
    const digest = createHash('sha256').update(bytes).digest('hex');
    if (digest !== ref.sha256) problems.push(`${key}: sha256 mismatch`);
  }
  return problems;
}

export function walkRelease(dir) {
  const out = [];
  const root = resolve(dir);
  const visit = (abs, rel) => {
    for (const name of readdirSync(abs).sort()) {
      const a = join(abs, name);
      const r = rel ? `${rel}/${name}` : name;
      const st = lstatSync(a);
      if (st.isSymbolicLink()) throw new Error(`symlink refused in release tree: ${r}`);
      if (!resolve(a).startsWith(root + sep)) throw new Error(`path escapes release tree: ${r}`);
      if (st.isDirectory()) visit(a, r);
      else if (st.isFile()) {
        if (!isSafeRelPath(r)) throw new Error(`unsafe file name in release tree: ${r}`);
        const ext = name.slice(name.lastIndexOf('.'));
        if (!ALLOWED_EXT.has(ext)) throw new Error(`unexpected file type in release tree: ${r}`);
        out.push(r);
      } else throw new Error(`non-regular file refused: ${r}`);
    }
  };
  visit(root, '');
  return out;
}

export function copyRelease(srcDir, siteDir) {
  const manifest = readManifest(srcDir);
  const problems = verifyManifest(srcDir, manifest);
  if (problems.length) throw new Error(`release ${manifest.release_id} failed verification:\n${problems.join('\n')}`);
  const dest = join(siteDir, 'data', manifest.release_id);
  const files = walkRelease(srcDir);
  for (const rel of files) {
    const to = join(dest, rel);
    mkdirSync(dirname(to), { recursive: true });
    copyFileSync(join(srcDir, rel), to);
  }
  return { releaseId: manifest.release_id, files: files.length };
}
