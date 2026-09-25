import { describe, expect, it } from 'vitest';
import { encodeId, isSafeKey, isSafeResourcePath, releaseUrl, parsePlayerIdParam } from '../../src/lib/ids';

describe('ids', () => {
  it('encodes public ids to resource keys', () => {
    expect(encodeId('legacy:daicos_nick_03012003')).toBe('legacy__daicos_nick_03012003');
    expect(encodeId('demo:2026:r01:a-b')).toBe('demo__2026__r01__a-b');
  });
  it('accepts only safe keys', () => {
    expect(isSafeKey('legacy__demo_player_a1')).toBe(true);
    expect(isSafeKey('a'.repeat(160))).toBe(true);
    expect(isSafeKey('a'.repeat(161))).toBe(false);
    for (const bad of ['', '../x', '..', 'a/b', '.hidden', 'a b', 'a%2f', 'x..y', '<script>', 'é']) {
      expect(isSafeKey(bad), bad).toBe(false);
    }
  });
  it('parses player id params from either id or key form', () => {
    expect(parsePlayerIdParam('legacy:demo_player_a1')).toEqual({ id: 'legacy:demo_player_a1', key: 'legacy__demo_player_a1' });
    expect(parsePlayerIdParam('legacy__demo_player_a1')).toEqual({ id: 'legacy:demo_player_a1', key: 'legacy__demo_player_a1' });
    expect(parsePlayerIdParam('../../etc/passwd')).toBeNull();
    expect(parsePlayerIdParam(null)).toBeNull();
    expect(parsePlayerIdParam('')).toBeNull();
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
