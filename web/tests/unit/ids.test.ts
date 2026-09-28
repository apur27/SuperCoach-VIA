import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeKey, encodeId, isSafeKey, isSafeResourcePath, releaseUrl, parsePlayerIdParam } from '../../src/lib/ids';

describe('ids', () => {
  it('encodes public ids to resource keys', () => {
    const canonical = encodeId('legacy:daicos_nick_03012003');
    expect(canonical.startsWith('k.')).toBe(true);
    expect(canonical.includes(':')).toBe(false);
    expect(canonical).not.toBe('legacy__daicos_nick_03012003');
    expect(decodeKey(canonical)).toBe('legacy:daicos_nick_03012003');
    expect(encodeId('a__b')).not.toBe(encodeId('id.a_x_b'));
    expect(decodeKey(encodeId('a__b'))).toBe('a__b');
    expect(decodeKey(encodeId('id.a_x_b'))).toBe('id.a_x_b');
    expect(decodeKey(encodeId('legacy:a___b'))).toBe('legacy:a___b');
    expect(decodeKey('legacy__demo_player_a1')).toBe('legacy:demo_player_a1');
  });
  it('round-trips a spread of ids and refuses an oversized one', () => {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:_-.';
    let state = 0x12345678;
    const next = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state;
    };
    const seen = new Map<string, string>();
    for (let n = 0; n < 60; n++) {
      const len = (next() % 40) + 1;
      let id = '';
      for (let i = 0; i < len; i++) id += alphabet[next() % alphabet.length];
      if (id.includes('..')) continue;
      const key = encodeId(id);
      expect(key.length).toBeLessThanOrEqual(200);
      expect(decodeKey(key)).toBe(id);
      const prev = seen.get(key);
      if (prev !== undefined) expect(prev).toBe(id);
      seen.set(key, id);
    }
    expect(() => encodeId(`legacy:${'a'.repeat(180)}`)).toThrow(/alias/);
  });
  it('accepts only safe keys', () => {
    expect(isSafeKey('legacy__demo_player_a1')).toBe(true);
    expect(isSafeKey('a'.repeat(200))).toBe(true);
    expect(isSafeKey('a'.repeat(201))).toBe(false);
    for (const bad of ['', '../x', '..', 'a/b', '.hidden', 'a b', 'a%2f', 'x..y', '<script>', 'é']) {
      expect(isSafeKey(bad), bad).toBe(false);
    }
  });
  it('parses player id params from either id or key form', () => {
    const key = encodeId('legacy:demo_player_a1');
    expect(parsePlayerIdParam('legacy:demo_player_a1')).toEqual({ id: 'legacy:demo_player_a1', key });
    expect(parsePlayerIdParam('legacy__demo_player_a1')).toEqual({ id: 'legacy:demo_player_a1', key });
    expect(parsePlayerIdParam(key)).toEqual({ id: 'legacy:demo_player_a1', key });
    expect(parsePlayerIdParam('../../etc/passwd')).toBeNull();
    expect(parsePlayerIdParam(null)).toBeNull();
    expect(parsePlayerIdParam('')).toBeNull();
    const legacyMatch = 'm__2025__r01__a-b';
    expect(parsePlayerIdParam(legacyMatch)).toEqual({
      id: 'm:2025:r01:a-b',
      key: encodeId('m:2025:r01:a-b'),
    });
    expect(parsePlayerIdParam(legacyMatch)?.key).not.toBe(legacyMatch);
  });
  it('match pages resolve a legacy colon-to-underscore bookmark through the same parser', () => {
    const src = readFileSync(new URL('../../src/islands/MatchView.tsx', import.meta.url), 'utf8');
    expect(src).toContain('parsePlayerIdParam');
    expect(src).not.toContain("raw.includes(':')");
  });
  it('rejects uncontained resource paths', () => {
    expect(isSafeResourcePath('players/legacy__x.json')).toBe(true);
    expect(isSafeResourcePath('matches/2026/index.json')).toBe(true);
    for (const bad of ['/abs.json', '../x.json', 'a/../b.json', 'https://evil.test/x.json', '//evil.test/x', 'a\\b.json', '', 'a//b.json', 'a/./b.json', 'a?b=1', 'a#b', 'a%2e%2e/b']) {
      expect(isSafeResourcePath(bad), bad).toBe(false);
    }
  });
  it('builds release-scoped same-origin urls', () => {
    expect(releaseUrl('/', 'r1', 'overview.json')).toBe('/data/r1/overview.json');
    expect(releaseUrl('/SuperCoach-VIA/', 'r1', 'players/index.json')).toBe('/SuperCoach-VIA/data/r1/players/index.json');
    expect(releaseUrl('/SuperCoach-VIA', 'r1', 'x.json')).toBe('/SuperCoach-VIA/data/r1/x.json');
    expect(() => releaseUrl('/', 'r1', '../x.json')).toThrow();
    expect(() => releaseUrl('/', '../r1', 'x.json')).toThrow();
  });
});
