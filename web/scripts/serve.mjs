// Minimal static server for the production build (preview + e2e). Binds 127.0.0.1 only.
// Usage: node scripts/serve.mjs [--dir dist] [--base /] [--port 4321]
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve, sep } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const root = resolve(args.dir ?? 'dist');
const base = (args.base ?? '/').replace(/\/?$/, '/');
const port = Number(args.port ?? 4321);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.csv': 'text/csv; charset=utf-8',
  '.zip': 'application/zip', '.md': 'text/markdown; charset=utf-8', '.txt': 'text/plain; charset=utf-8', '.ico': 'image/x-icon',
};

function send(res, status, file) {
  res.writeHead(status, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', 'x-content-type-options': 'nosniff', 'cache-control': 'no-cache' });
  createReadStream(file).pipe(res);
}

createServer((req, res) => {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://127.0.0.1').pathname);
  } catch {
    res.writeHead(400).end('bad request');
    return;
  }
  const notFound = () => {
    const page = join(root, '404.html');
    if (existsSync(page)) send(res, 404, page);
    else res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
  };
  if (!pathname.startsWith(base)) {
    if (pathname === base.slice(0, -1)) {
      res.writeHead(301, { location: base }).end();
      return;
    }
    return notFound();
  }
  const rel = pathname.slice(base.length);
  const target = normalize(join(root, rel));
  if (target !== root && !target.startsWith(root + sep)) return notFound();
  let file = target;
  if (existsSync(file) && statSync(file).isDirectory()) {
    if (!pathname.endsWith('/')) {
      res.writeHead(301, { location: `${pathname}/` }).end();
      return;
    }
    file = join(file, 'index.html');
  }
  if (!existsSync(file) || !statSync(file).isFile()) return notFound();
  send(res, 200, file);
}).listen(port, '127.0.0.1', () => console.log(`serving ${root} at http://127.0.0.1:${port}${base}`));
