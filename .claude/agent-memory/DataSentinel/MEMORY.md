# DataSentinel Memory Index

- [Hall of Fame stat-leaders refresh pattern](project_hof_refresh_pattern.md) — JSON ground truth in docs/hall-of-fame/_stat_leaders.json; hub + 7 sub-pages refreshed 2026-06-22
- [Canonical games metric](feedback_canonical_games_metric.md) — career games = max(rowcount, games_played.max()), NOT len(df); naive rowcount induces false FAILs (relocated from Gaffer, A-05)
- [Player CSVs not chronological](feedback_player_csv_not_chronological.md) — sort_values(['year','round']) before ANY last-N slice; rows are not date-ordered (relocated from Skeptic, A-05)
- [Audit URL-collision false-WARNINGs](project_audit_url_collision_fp.md) — career-total WARNINGs often false: URL builder discards DOB, collides same-name players; triage by DOB-stamped filename (relocated from Gaffer, A-05)
