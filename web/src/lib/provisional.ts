/** Public source-audit confidence is bound to its snapshot, separately from freshness and release validation. */
export const PROVISIONAL_SNAPSHOT_ID = 'sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0';
export const CORRECTED_SNAPSHOT_ID = 'sha256:b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4';
const readmeUrl = 'https://github.com/apur27/SuperCoach-VIA/blob/main/README.md#data-status';
interface AuditEvidence { label: string; snapshotId: string; reportSha256: string }
interface AuditMetadata {
  verdict: 'FAIL' | 'UNKNOWN' | 'UNAUDITED';
  banner: string;
  title: string;
  snapshotId: string;
  snapshotCreatedAt: string | null;
  evidence: readonly AuditEvidence[];
  readmeUrl: string;
  reportsUrl: string | null;
  reportsLabel: string | null;
  coverageUrl: string | null;
}
const audit: AuditMetadata = {
  verdict: 'FAIL',
  banner: 'Provisional data — known audit failures',
  title: 'Known source audit failures',
  snapshotId: PROVISIONAL_SNAPSHOT_ID,
  snapshotCreatedAt: '2026-09-29T20:38:16.571555Z',
  evidence: [{ label: 'Recorded source audit', snapshotId: PROVISIONAL_SNAPSHOT_ID, reportSha256: '58eff517a29d30b932087f919a31d36567e8be330301ecff53feb06494d8db28' }],
  readmeUrl,
  reportsUrl: 'https://github.com/apur27/SuperCoach-VIA/blob/main/docs/hall-of-fame/provisional/README.md',
  reportsLabel: 'Provisional Hall of Fame reports',
  coverageUrl: null,
};
const correctedAudit: AuditMetadata = {
  verdict: 'UNKNOWN',
  banner: 'Provisional data — source audit UNKNOWN',
  title: 'Unresolved source audit evidence',
  snapshotId: CORRECTED_SNAPSHOT_ID,
  snapshotCreatedAt: '2026-10-05T20:43:39.205629Z',
  evidence: [
    { label: 'Full parent audit (UNKNOWN)', snapshotId: 'sha256:f1abd8c2b7f8d6b3812e73f91a4403add1844f3ad9037139e7dc373a0ed91703', reportSha256: '81ac70ba8aa25110119a91779d2c2df91088b2f0f87408db3546de8109a4509c' },
    { label: 'Candidate audit, season 2026 only (PASS)', snapshotId: CORRECTED_SNAPSHOT_ID, reportSha256: '6fecac8e1c8eb804774eb70a5f9d03bd5448ee86d43d1253472ca0f1ea7d9c2a' },
  ],
  readmeUrl,
  reportsUrl: 'https://github.com/apur27/SuperCoach-VIA/blob/main/docs/reviews/AFLTABLES_RECONCILIATION_RUN.md',
  reportsLabel: 'Source audit run record',
  coverageUrl: 'https://github.com/apur27/SuperCoach-VIA/blob/main/docs/reviews/evidence/afltables-candidate-source-coverage-20261007.json',
};
export function provisionalAudit(snapshotId: string, demo: boolean): AuditMetadata | null {
  if (demo) return null;
  if (snapshotId === PROVISIONAL_SNAPSHOT_ID) return audit;
  if (snapshotId === CORRECTED_SNAPSHOT_ID) return correctedAudit;
  return {
    verdict: 'UNAUDITED',
    banner: 'Provisional data — source audit unavailable',
    title: 'Source audit unavailable',
    snapshotId,
    snapshotCreatedAt: null,
    evidence: [],
    readmeUrl,
    reportsUrl: null,
    reportsLabel: null,
    coverageUrl: null,
  };
}
