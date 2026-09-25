import { describe, expect, it } from 'vitest';
import { toCsv } from '../../src/lib/csv';

describe('csv export', () => {
  it('quotes, neutralizes formulas, keeps numbers and blanks nulls', () => {
    const csv = toCsv(['name', 'value', 'note'], [['=HYPERLINK("x")', 12.5, null], ['a,b', -3, '+1'], ['@cmd', 0, 'plain "q"']]);
    expect(csv).toBe('name,value,note\n"\'=HYPERLINK(""x"")",12.5,\n"a,b",-3,\'+1\n\'@cmd,0,"plain ""q"""\n');
  });
});
