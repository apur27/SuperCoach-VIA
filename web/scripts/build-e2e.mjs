// Builds the production site twice (base "/" and "/SuperCoach-VIA/") for e2e tests.
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const variants = [
  { name: 'root', base: '/' },
  { name: 'sub', base: '/SuperCoach-VIA/' },
];
for (const v of variants) {
  const outDir = resolve('.e2e-dist', v.name);
  const r = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], {
    stdio: 'inherit',
    env: { ...process.env, SCVIA_PUBLIC_BASE: v.base, SCVIA_OUT_DIR: outDir },
  });
  if (r.status !== 0) process.exit(r.status ?? 1);
}
