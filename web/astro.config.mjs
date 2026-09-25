// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  copyRelease, isSafeRelPath, readManifest, releaseDirFromEnv, retainedDirsFromEnv, verifyManifest,
} from './integrations/release-tree.mjs';

const base = process.env.SCVIA_PUBLIC_BASE || '/';
if (!/^\/([A-Za-z0-9._-]+\/?)*$/.test(base)) throw new Error(`invalid SCVIA_PUBLIC_BASE: ${base}`);
const site = process.env.SCVIA_PUBLIC_SITE || undefined;
if (site && !/^https:\/\/[A-Za-z0-9.-]+(\/.*)?$/.test(site) && !/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(site)) {
  throw new Error(`invalid SCVIA_PUBLIC_SITE: ${site}`);
}
const releaseDir = releaseDirFromEnv();
// Pages read the release at build time; pass the resolved absolute path (bundling breaks import.meta.url).
process.env.SCVIA_RELEASE_DIR_RESOLVED = releaseDir;
const manifest = readManifest(releaseDir);
const problems = verifyManifest(releaseDir, manifest);
if (problems.length) throw new Error(`release manifest verification failed:\n${problems.join('\n')}`);

/** @returns {import('astro').AstroIntegration} */
function releaseData() {
  return {
    name: 'scvia-release-data',
    hooks: {
      'astro:server:setup': ({ server }) => {
        // Dev only: serve the release tree at <base>data/<release_id>/.
        const prefix = `${base.replace(/\/?$/, '/')}data/${manifest.release_id}/`;
        server.middlewares.use((req, res, next) => {
          const url = (req.url ?? '').split('?')[0] ?? '';
          if (!url.startsWith(prefix)) return next();
          const rel = decodeURIComponent(url.slice(prefix.length));
          const file = join(releaseDir, rel);
          if (!isSafeRelPath(rel) || !existsSync(file) || !statSync(file).isFile()) {
            res.statusCode = 404;
            return res.end('not found');
          }
          res.setHeader('content-type', rel.endsWith('.json') ? 'application/json' : 'application/octet-stream');
          createReadStream(file).pipe(res);
        });
      },
      'astro:build:done': ({ dir, logger }) => {
        const out = fileURLToPath(dir);
        const main = copyRelease(releaseDir, out);
        logger.info(`copied release ${main.releaseId} (${main.files} files)`);
        for (const d of retainedDirsFromEnv()) {
          const r = copyRelease(d, out);
          logger.info(`copied retained release ${r.releaseId} (${r.files} files)`);
        }
      },
    },
  };
}

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  output: 'static',
  outDir: process.env.SCVIA_OUT_DIR ? resolve(process.env.SCVIA_OUT_DIR) : './dist',
  build: { format: 'directory', inlineStylesheets: 'never' },
  integrations: [react(), releaseData()],
  devToolbar: { enabled: false },
  markdown: { syntaxHighlight: false },
  prefetch: false,
  security: {
    checkOrigin: true,
    csp: {
      algorithm: 'SHA-256',
      directives: [
        "default-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "connect-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "form-action 'self'",
        "manifest-src 'self'",
        "worker-src 'none'",
      ],
    },
  },
  vite: {
    build: { assetsInlineLimit: 0 },
  },
});
