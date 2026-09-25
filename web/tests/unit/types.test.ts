// Compile-time drift checks (enforced by `npm run check` / tsc): nullability and enums
// in the generated types must match the schemas.
import { describe, expectTypeOf, it } from 'vitest';
import type { Freshness, StatValue, PredictionRow, MatchSummary } from '../../src/lib/contracts';

describe('generated types', () => {
  it('keep nullable fields nullable and enums closed', () => {
    expectTypeOf<Freshness['source_checked_at']>().toEqualTypeOf<string | null>();
    expectTypeOf<Freshness['stale']>().toEqualTypeOf<boolean>();
    expectTypeOf<Freshness['dataset_status']>().toEqualTypeOf<'legacy_unverified' | 'verified' | 'partial' | 'demo'>();
    expectTypeOf<StatValue['total']>().toEqualTypeOf<number | null>();
    expectTypeOf<StatValue['observed_games']>().toEqualTypeOf<number>();
    expectTypeOf<PredictionRow['interval_low']>().toEqualTypeOf<number | null>();
    expectTypeOf<PredictionRow['origin']>().toEqualTypeOf<'prospective' | 'replay' | 'legacy_unknown'>();
    expectTypeOf<MatchSummary['match_date']>().toEqualTypeOf<string | null>();
  });
});
