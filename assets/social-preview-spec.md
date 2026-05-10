# Social preview image specification

GitHub displays a **1280×640 px** image when this repo is shared on Twitter/X, LinkedIn, Slack, or in a GitHub Open Graph card. The image lives in **GitHub repo settings → Social preview**.

This file describes what that image should contain. The PNG itself is intentionally not in this repo yet - generation requires graphic tooling (Figma, Canva, or matplotlib + Pillow). A future automated step in `scripts/` may produce it from this spec.

---

## Constraints

- **Dimensions**: 1280×640 px (2:1 ratio). Anything else and GitHub crops or rejects it.
- **File size**: under 1 MB. PNG preferred; JPG acceptable.
- **Safe zone**: keep critical content within the central 1100×500 px - some social previews crop the edges.
- **Text legibility**: must read at thumbnail size (320×160 px). Headlines no smaller than 64 px.

## Brand alignment

The repo banner (`assets/banner.svg`, `assets/readme_banner.png`) is the visual reference. Match its aesthetic: dark-navy / charcoal background with a single warm-accent colour (red or orange), clean sans-serif type, and analytics-flavoured iconography (charts, heat maps, pitch overlays).

## Content - what the image should communicate

The viewer has 1.5 seconds. They should understand:

1. **It's about AFL** - the visual reference (a football, a ground silhouette, a club guernsey palette) makes that obvious.
2. **It's about data and predictions** - a chart, a small table, or "predict + analyse" iconography signals this.
3. **It's a public open-source project** - the GitHub URL or `apur27/SuperCoach-VIA` text is visible.

## Recommended layout

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [Repo banner artwork - football + analytics motif]         │
│                                                             │
│  AFL SuperCoach VIA                                         │
│  125+ years of AFL data, weekly predictions, no code needed │
│                                                             │
│                                                             │
│  github.com/apur27/SuperCoach-VIA          MIT · Python 3.10│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Headline copy (must include one of these)

- `AFL SuperCoach VIA`
- `125+ years of AFL data → weekly predictions`
- `Open-source AFL prediction & analytics`

### Sub-headline (pick one)

- `MAE 4.1 disposals · 68% within 5 · LightGBM ensemble`
- `Predictions, Hall of Fame, and historical insights - no code required for fans`
- `Weekly auto-refresh · GroupKFold backtest · era-fair rankings`

## Typography

- Headline: 96-128 px bold sans (Inter, Helvetica Neue, or repo banner font).
- Sub-head: 48-56 px medium.
- URL footer: 32-40 px regular.

## Colour palette

- Background: `#0E1B2C` (deep navy) or repo banner's exact dark.
- Primary accent: `#E63946` (warm red) or banner's accent.
- Secondary: white `#FFFFFF` for headline, `#A8B2C1` (light grey) for sub-head.
- Avoid pure black. Avoid more than three accent colours.

## Generation methods

When someone has time to make this:

### Option A - Figma / Canva (manual, 30 minutes)

1. Open the repo banner SVG in Figma.
2. Resize / re-crop to 1280×640.
3. Add the headline and footer text.
4. Export as PNG.
5. Upload via **GitHub repo → Settings → Social preview → Edit**.

### Option B - matplotlib + Pillow (scripted, future)

A `scripts/generate_social_preview.py` can be added to:
1. Render the banner SVG to PNG via `cairosvg` or `svglib`.
2. Composite the headline / footer text via Pillow.
3. Save to `assets/social_preview.png`.
4. Repo owner uploads it manually one time (GitHub does not auto-pick up files from the repo).

### Option C - off-the-shelf preview generator

Several tools (e.g., GitHub's own social preview, opengraph.xyz) generate a default preview from the README banner. Acceptable as an interim until a custom one is made.

---

## Acceptance checklist

Before uploading, verify:

- [ ] Exactly 1280×640 px
- [ ] Under 1 MB
- [ ] Headline readable at 320×160 px thumbnail
- [ ] Repo URL or `apur27/SuperCoach-VIA` text visible
- [ ] AFL theme is visually unambiguous within 1.5 seconds
- [ ] No personal photos, no logos of real AFL clubs (avoids trademark/IP issues - the banner artwork is custom)
- [ ] Test the upload by sharing a repo link in a private Slack or Twitter draft and confirming the preview renders correctly

---

## Related

- [`assets/banner.svg`](banner.svg) - the visual reference
- [`assets/readme_banner.png`](readme_banner.png) - PNG render of the banner
- [README](../README.md) - source of headline copy
