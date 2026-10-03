import assert from 'node:assert/strict';
import { test } from 'node:test';
import { aliasTarget } from '../public/path-map.mjs';

for (const [path, search, hash, expected] of [
  ['/supercoach-via', '', '', '/SuperCoach-VIA/'],
  ['/supercoach-via/', '', '', '/SuperCoach-VIA/'],
  ['/SUPERCOACH-VIA/player/', '?id=k.A%2FB%3D&label=Ben+Smith', '#career-h', '/SuperCoach-VIA/player/?id=k.A%2FB%3D&label=Ben+Smith#career-h'],
  ['/Supercoach-Via/articles/A%20B%2fc/', '?q=%23&next=https%3A%2F%2Fevil.example%2F', '#part%2Fone', '/SuperCoach-VIA/articles/A%20B%2fc/?q=%23&next=https%3A%2F%2Fevil.example%2F#part%2Fone'],
  ['/supercoach-via//evil.example/path', '?next=//evil.example', '#https://evil.example', '/SuperCoach-VIA//evil.example/path?next=//evil.example#https://evil.example'],
  ['/supercoach-via/player/', '?next=javascript:alert(1)', '', '/SuperCoach-VIA/player/?next=javascript:alert(1)'],
]) {
  test(`maps alias ${path} while preserving suffix, query and fragment`, () => {
    assert.equal(aliasTarget(path, search, hash), expected);
    const target = new URL(expected, 'https://apur27.github.io');
    assert.equal(target.origin, 'https://apur27.github.io');
    assert.ok(target.pathname.startsWith('/SuperCoach-VIA/'));
  });
}

for (const path of [
  '/SuperCoach-VIA', '/SuperCoach-VIA/', '/SuperCoach-VIA/player/',
  '/', '/another-project/', '/supercoach-via-extra/', '/supercoach-via.example/',
  '/nested/supercoach-via/', '/supercoach%2Dvia/', '//evil.example/supercoach-via/',
  'https://evil.example/supercoach-via/', 'javascript:alert(1)',
  '/supercoach-via/../elsewhere/', '/supercoach-via/%2e%2e/elsewhere/',
  '/supercoach-via/.%2E/elsewhere/', '/supercoach-via/\\..\\elsewhere/',
]) {
  test(`does not redirect canonical, unrelated or escaping path ${path}`, () => {
    assert.equal(aliasTarget(path, '?id=keep', '#keep'), null);
  });
}

test('rejects malformed query and hash arguments instead of creating a different path', () => {
  assert.equal(aliasTarget('/supercoach-via/', '/other/', ''), null);
  assert.equal(aliasTarget('/supercoach-via/', '', '?other'), null);
});
