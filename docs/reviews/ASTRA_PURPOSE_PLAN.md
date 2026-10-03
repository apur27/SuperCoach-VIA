# Homepage purpose: Astra follow-up plan

Read the complete “Why this repo exists” section of the main README, lines 317–end, on 3 October 2026. The current homepage explains features but omits the personal and community reasons the project exists. This is a substantive omission, not a request for more feature copy.

## Ordered implementation

1. Keep the H1, player search, direct destinations and provisional notices. Replace the current feature-only lede with the short purpose sentence below, followed by a prominent ordinary link to `#why-this-project`. Keep the data-status sentence visible. This makes purpose apparent before scrolling, without requiring the reader to open another page.
2. Add an always-visible “Why this project exists” section immediately after the intro/actions/snapshot notes, before the season label and Recent results. Use the exact story below. This puts the README’s motivation ahead of the statistical report while search stays above it. Do not put this story inside a closed disclosure or below articles.
3. Present the story as the existing white/dark surface with a clear H2, then “From the project creator” as a muted attribution. Limit prose to about 70ch, left align, retain normal body size and comfortable paragraph spacing. Use two columns only if it preserves the reading order; simplest and preferred is one prose column inside the full-width section. No portrait, invented quotation or decorative image is needed.
4. Preserve the rest of the homepage’s order and behaviour. A separate About page is unnecessary: this bounded story represents all material motivations in the README directly on the homepage. No numeric/data/pipeline/schema/Claude changes.

## Exact introductory copy

Replace the existing lede with:

> A noncommercial AFL project, inspired by years of SuperCoach with friends and gratitude for the people who make the game welcoming.

Add immediately after it, as a plain visible link:

> Why this project exists

Link target: `#why-this-project`. Use a stable section ID so pointer, keyboard and no-JS navigation all work. The existing H1 and destination labels already explain the practical features; no extra feature paragraph is necessary.

## Exact story copy

### Why this project exists

*From the project creator*

This project is about the game, its history and the people it brings together. It is noncommercial, has no affiliation with gambling services and is not intended to encourage betting.

I have played SuperCoach with the same group for over a decade. Arguments about who the better player really was, and Sunday-night lineup tweaks, grew into a project for exploring the game through data.

It is also a thank you to the friends and colleagues who introduced me to AFL and SuperCoach, explained clearances and ruck craft, and helped me understand a scoreline.

A special thank you goes to the families, coaches and community of Cranbourne Junior Football Club, who welcomed my son and taught him the game in the right spirit. The volunteers giving their time on cold mornings and the families standing on the boundary are part of what makes the game matter.

For me, that welcome is also part of AFL’s contribution to a multicultural Australia. A new Australian joining a junior football club and being welcomed onto a team is a meaningful form of belonging.

This work honours the players of the past, present and future, and the generations of fans they have brought together. Their careers are the reason for keeping the record.

## Grounding and editorial boundaries

Every autobiographical detail and motivation above comes from the README section: over a decade with the same SuperCoach group; debates and lineup tinkering; friends/colleagues teaching the game; the creator’s son welcomed at Cranbourne Junior Football Club; volunteer coaches and boundary-side families; multicultural belonging; homage to players across generations. The first-person voice is attributed explicitly to the project creator. Do not introduce the creator’s name, migration background, location, career, family details, club endorsement or official affiliation. The closing sentence expresses motivation and does not claim that the dataset is complete or verified.

## Acceptance checks

- The homepage’s first screen explains the community purpose and has a clear “Why this project exists” link. Search stays above the story; at 390px its top remains within a 900px viewport if practical without shrinking typography or notices.
- Keyboard activation of the intro link reaches the visible story heading. The story is present with JavaScript disabled.
- All seven source themes remain visible: personal SuperCoach origins; friends/colleagues; Cranbourne club and son; volunteering; multicultural belonging; honouring players; noncommercial/no gambling purpose.
- No autobiographical facts beyond the README; no change to player numbers, coverage, forecasts or warning text.
- Check 320/390/768/1440px in light and dark: no document overflow, comfortable prose width, no awkward heading collision or added two-column mobile layout.
- Provisional banner, snapshot notes and unavailable forecast remain visible and accurate. Existing search/direct destinations still work.
- Focused tests for content presence, native anchor and no-JS visibility; parent chooses the appropriate existing regression checks. Astra inspects actual rebuilt desktop/mobile screenshots before final signoff.

Scope: `web/src/pages/index.astro`, a small scoped/shared style rule only if existing styles cannot provide a 70ch prose measure, and focused tests. This follow-up is content presentation; it requires no new dependency, route or data generation logic.
