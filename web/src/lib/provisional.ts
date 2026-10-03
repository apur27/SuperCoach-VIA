/** Public audit context is pinned to its input snapshot, never inferred from release validation. */
export const PROVISIONAL_SNAPSHOT_ID = 'sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0';
const audit = {
  snapshotId: PROVISIONAL_SNAPSHOT_ID,
  reportSha256: '58eff517a29d30b932087f919a31d36567e8be330301ecff53feb06494d8db28',
  snapshotCreatedAt: '2026-09-29T20:38:16.571555Z',
  readmeUrl: 'https://github.com/apur27/SuperCoach-VIA/blob/main/README.md#data-status',
  reportsUrl: 'https://github.com/apur27/SuperCoach-VIA/blob/main/docs/hall-of-fame/provisional/README.md',
} as const;
export function provisionalAudit(snapshotId: string, demo: boolean) {
  return !demo && snapshotId === PROVISIONAL_SNAPSHOT_ID ? audit : null;
}
