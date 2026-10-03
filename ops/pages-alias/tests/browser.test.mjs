import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { chromium } from '../../../web/node_modules/@playwright/test/index.mjs';

const publicDir = new URL('../public/', import.meta.url);
const assets = new Map([
  ['/alias.mjs', ['alias.mjs', 'text/javascript']],
  ['/path-map.mjs', ['path-map.mjs', 'text/javascript']],
  ['/alias.css', ['alias.css', 'text/css']],
]);
let server;
let browser;
let origin;
before(async () => {
  server = createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (path === '/SuperCoach-VIA' || path.startsWith('/SuperCoach-VIA/')) {
      res.writeHead(200, { 'content-type': 'text/html' });
      return res.end('<!doctype html><html lang="en"><title>Project destination</title><body><h1>Project destination</h1></body></html>');
    }
    const asset = assets.get(path);
    if (asset) {
      res.writeHead(200, { 'content-type': asset[1] });
      return res.end(readFileSync(new URL(asset[0], publicDir)));
    }
    if (path === '/') {
      res.writeHead(200, { 'content-type': 'text/html' });
      return res.end(readFileSync(new URL('index.html', publicDir)));
    }
    const entry = path === '/supercoach-via' || path === '/supercoach-via/';
    res.writeHead(entry ? 200 : 404, { 'content-type': 'text/html' });
    res.end(readFileSync(new URL(entry ? 'supercoach-via/index.html' : '404.html', publicDir)));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ channel: 'chromium' });
});
after(async () => {
  await browser?.close();
  if (server) await new Promise((resolve) => server.close(resolve));
});

for (const [source, destination] of [
  ['/supercoach-via', '/SuperCoach-VIA/'],
  ['/supercoach-via/', '/SuperCoach-VIA/'],
  ['/supercoach-via/player/?id=k.A%2fB%3D&label=Ben+Smith#career-h', '/SuperCoach-VIA/player/?id=k.A%2fB%3D&label=Ben+Smith#career-h'],
  ['/SUPERCOACH-VIA/articles/A%20B/?next=https%3A%2F%2Fevil.example%2F#part%2Fone', '/SuperCoach-VIA/articles/A%20B/?next=https%3A%2F%2Fevil.example%2F#part%2Fone'],
]) {
  test(`browser maps ${source} to the same-origin project without losing URL bytes`, async () => {
    const page = await browser.newPage();
    const errors = [];
    const modules = [];
    page.on('response', (response) => {
      if (new URL(response.url()).pathname.endsWith('.mjs')) modules.push([new URL(response.url()).pathname, response.headers()['content-type']]);
    });
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => {
      if (/Content Security Policy|Refused to/.test(message.text())) errors.push(message.text());
    });
    await page.goto(`${origin}${source}`);
    await page.waitForURL(`${origin}${destination}`);
    assert.equal(page.url(), `${origin}${destination}`);
    assert.equal(await page.locator('h1').textContent(), 'Project destination');
    assert.deepEqual(errors, []);
    assert.deepEqual(modules.map(([path]) => path).sort(), ['/alias.mjs', '/path-map.mjs']);
    for (const [, mime] of modules) assert.match(mime, /^text\/javascript/);
    await page.close();
  });
}

for (const source of ['/?from=root#home', '/SuperCoach-VIA/player/?id=keep#keep', '/another-project/?id=keep#keep', '/supercoach-via-extra/']) {
  test(`browser leaves ${source} alone`, async () => {
    const page = await browser.newPage();
    await page.goto(`${origin}${source}`);
    await page.waitForLoadState('networkidle');
    assert.equal(page.url(), `${origin}${source}`);
    await page.close();
  });
}

test('entry and fallback offer an accessible plain link with JavaScript disabled', async () => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 320, height: 800 } });
  const page = await context.newPage();
  for (const [source, status] of [['/supercoach-via/', 200], ['/SUPERCOACH-VIA/player/?id=keep#keep', 404]]) {
    const response = await page.goto(`${origin}${source}`);
    assert.equal(response.status(), status);
    const link = page.getByRole('link', { name: 'continue to SuperCoach VIA' });
    assert.ok(await link.isVisible());
    assert.equal(await link.getAttribute('href'), '/SuperCoach-VIA/');
    await link.focus();
    assert.notEqual(await link.evaluate((el) => getComputedStyle(el).outlineStyle), 'none');
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth));
    await link.click();
    assert.equal(page.url(), `${origin}/SuperCoach-VIA/`);
  }
  await context.close();
});
