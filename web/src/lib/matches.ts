import type { MatchSummary, TeamScore } from './contracts';

export function teamScoreText(t: TeamScore): string {
  if (t.score === null) return 'score not recorded';
  if (t.goals === null || t.behinds === null) return String(t.score);
  return `${t.goals}.${t.behinds} (${t.score})`;
}

const STATUS: Record<MatchSummary['status'], string> = {
  scheduled: 'Scheduled', in_progress: 'In progress (as of last snapshot)', complete: 'Complete', postponed: 'Postponed', cancelled: 'Cancelled', unknown: 'Status unknown',
};
export function statusLabel(s: MatchSummary['status']): string {
  return STATUS[s];
}

export function resultLine(m: MatchSummary): string {
  if (m.status === 'scheduled') return 'Not yet played';
  if (m.status === 'postponed') return 'Postponed';
  if (m.status === 'cancelled') return 'Cancelled';
  if (m.home.score === null || m.away.score === null) return 'Result not recorded';
  if (m.status !== 'complete') return statusLabel(m.status);
  if (m.home.score === m.away.score) return 'Draw';
  const [w, l] = m.home.score > m.away.score ? [m.home, m.away] : [m.away, m.home];
  return `${w.name} won by ${(w.score ?? 0) - (l.score ?? 0)}`;
}

export function isReplay(m: MatchSummary): boolean {
  return m.replay_occurrence > 1;
}

const REASONS: Record<string, string> = {
  no_valid_future_fixture: 'No valid future fixture: there are no scheduled matches after the latest completed match, so no forecast is published.',
  season_complete: 'The season is complete; no future matches are scheduled.',
  no_model: 'No promoted model is available for this release.',
  expired: 'The forecast targets have already been played.',
};
export function reasonText(reason: string | null | undefined): string {
  if (!reason) return 'No reason was recorded.';
  return REASONS[reason] ?? reason;
}
