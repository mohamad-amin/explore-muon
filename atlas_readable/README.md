# Readable Gauss-Newton Atlas

A re-designed, presentation-only edition of the interactive "Gauss-Newton Atlas"
(`logs/muon_spectra/second_order_audit_20260926/make_atlas.py` + `atlas_template.html`). It shows the same 14 views,
built from the same numbers, laid out for reading: full-width pages, large plots, big type, legends that never cover data,
labeled reference lines, one consistent colorblind-safe color per optimizer, and a "What to look for" caption under every chart.

Published artifact (private): https://claude.ai/artifact/1r6g3Sw7GVR8fZxxy4YzcN, first published 2026-09-28 13:45 CDT.
The original Atlas artifact (https://claude.ai/artifact/8tZX3HjvacAn7ZhDJSxC8x) is left untouched.

## Files

| file | what it is |
|---|---|
| `build_readable_atlas.py` | builder: runs the original builder read-only, takes its payload unchanged, fills the template |
| `readable_template.html` | the page (CSS, layout, all 14 view renderers); placeholders `__ATLAS_DATA__`, `__ATLAS_TEXT__`, `__ATLAS_META__` |
| `gauss_newton_atlas_readable.html` | the built page (about 2.7 MB, of which 2.48 MB is the payload, as of 2026-09-29) |
| `tools/extract_text.mjs` | pulls the view explanations and the Principles out of the original `atlas_template.html` at build time |
| `tools/check_keys.mjs` | static check: which payload keys each view reads, and whether they exist |
| `tools/screenshot.mjs` | headless-Chromium check: renders every view, screenshots, console errors, overflow, missing-key reads |
| `tools/tile.py` | slices full-page screenshots into viewport-sized tiles for review |
| `tools/palette_search.mjs`, `tools/palette_refine.mjs` | the searches used to pick the optimizer colors (validated afterwards with the dataviz palette checker) |
| `tools/palette_variants.mjs` | the search for the four colors of non-optimizer arms (variants of PD), kept away from the optimizer hues |
| `build/original_atlas.html` | the original builder's output from the last build (the payload source) |
| `build/screens/` | last full verification run: full-page PNGs at 1600 px, 390 px and 1600 px dark, plus `report*.json` |
| `build/screens_edge0929/` | the 2026-09-29 Edge verification run (Edge and Principles at 1600 and 390 px, Edge dark) |

## Provenance and data fidelity

- `build_readable_atlas.py` runs the original `make_atlas.py` as a subprocess with `python -B` (so no bytecode is
  written into the audit folder) and output `build/original_atlas.html`. Nothing in the audit folder or in `research/` is written.
- The JSON inside `<script id="atlas-data">` is copied **byte for byte** into the new page; the builder re-extracts it
  from the output and fails if it differs. It also rejects NaN/Infinity (which `JSON.parse` cannot read).
- No measurement is recomputed. The page only selects, labels and plots values from the payload, exactly as the original
  renderers did (same medians over layers, same ratios such as score ÷ exact-GN score, same filters, e.g. step 50 is still
  left out of the midpoint chart).
- The view explanations and the Principles are re-extracted from the original `atlas_template.html` on every build, so
  the wording stays current. A few phrases that describe the original page's colors or panel positions are adjusted
  (e.g. "teal early, orange late" becomes "lighter early, darker late"; "Left/Middle/Right" becomes "First/Second/Third").
- The page footer records the build time (Chicago), the original builder's path, and the payload's size and SHA-256
  (the builder's and template's hashes are in the page's `atlas-meta` JSON).

## Rebuild

```bash
cd /share/data/dl-theory/amin/projects/explore_muon
.venv/bin/python -B atlas_readable/build_readable_atlas.py            # runs the original builder (15-50 s), writes the page
.venv/bin/python -B atlas_readable/build_readable_atlas.py --reuse-original   # reuse build/original_atlas.html (template edits only)
```

Needs `node` on PATH for the text extraction (without it the page falls back to its built-in copy of the texts).
To update the published artifact, republish `atlas_readable/gauss_newton_atlas_readable.html` to the same artifact URL.

## Verify

```bash
cd /share/data/dl-theory/amin/projects/explore_muon/atlas_readable
node tools/check_keys.mjs gauss_newton_atlas_readable.html
export LD_LIBRARY_PATH=$PWD/.syslibs/root/usr/lib/x86_64-linux-gnu PLAYWRIGHT_BROWSERS_PATH=$PWD/.pw-browsers TMPDIR=$PWD/.tmp
node tools/screenshot.mjs --widths 1600,390            # add --dark for the dark theme, --views a,b to limit
../.venv/bin/python -B tools/tile.py build/screens/1600 1000
```

The browser tooling lives only in this folder: `node_modules/` (playwright 1.49.1, plotly.js-dist-min 2.35.2, 10 MB),
`.pw-browsers/` (Chromium 131, 465 MB) and `.syslibs/` (Debian bookworm libnss3, libnspr4, libatk, libatk-bridge,
libatspi, libxcomposite unpacked with `apt-get download` + `dpkg -x`, because the host lacks them and there is no root).
None of it is needed to build or view the page; delete those three folders to reclaim the space.

## What changed (all views)

- **Width and layout.** Full viewport width up to 1900 px; views in a left sidebar (a top-bar picker below 1100 px).
  At most two plots per row, and two columns only while each plot stays at least 600 px wide (602 x 540 px at a
  1600 px window); dense panels span the whole row. Plots are 520 to 820 px tall on desktop and at least 420 px on phones.
  Phones get a 16 px gutter and no horizontal scroll.
- **Type.** Tick labels 14 px (13 px on phones), axis titles 16 px, panel titles 19 px, legends 14.5 px, hover text 14 px.
- **Legends** are HTML rows above each plot, so they never cover data; click an entry to hide or show its series.
  Where two things are encoded (e.g. optimizer by color and batch by line style), the legend is split into those two keys.
- **Reference lines** are dotted and labeled ("1 = K-FAC exact", "our batch (2048 seq.)", ...). The label goes to whichever
  end of the line is clear of data, or into the legend when neither is (always for bar charts).
- **Log axes** get explicit 1-2-5 (or powers-of-ten) ticks with full-size labels; Plotly's own 2 and 5 labels are tiny.
- **Color.** One color per optimizer everywhere: Muon blue, PD rust, SOAP-Muon green, S∘PD violet, TS magenta; exact /
  damped GN ink, PD in GN geometry gold, K-FAC/EKFAC slate, gradient and baselines grey. The five optimizer colors pass the
  dataviz checker's all-pairs CVD (protan/deutan) and normal-vision separation, lightness band, chroma and 3:1 contrast
  checks in both themes (light on #ffffff, dark on #151c23). Training steps, curvature classes and similar ordered series
  use a light-to-dark ramp of the owning optimizer's color, explained in the legend. Lines are 2 px or more with markers.
- **Labels.** Run names such as `PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada` are shown as "PD α½ · lr 0.02 · β 0.9", listing only
  what differs between runs of the same optimizer; the raw name is in the hover and under each table row.
- **Hover** shows the series name, the x and y values with their meaning, and formatted numbers.
- **Behavior.** Only the active view is built, and its charts are plotted as they approach the viewport. The view is kept
  in the URL hash (`#gap`, `#steprule`, ...) and, with the controls, in localStorage. Plot colors follow the light/dark theme.
- **Captions.** Each view opens with the original explanation (the Gap map caveat as a callout); every chart has a one- or
  two-sentence "What to look for" caption written for this edition from the original texts and card subtitles.
- **Non-optimizer arms** (e.g. coordination variants and staged branches of PD) use a separate four-color palette (indigo,
  coral, olive, lilac) that passes the same all-pairs checks in both themes and is kept away from the optimizer hues.
- **Heatmaps** use a theme-aware diverging scale (blue, neutral gray, red) with a labeled color bar, square cells and
  readable bin labels; on phones the color bar sits above the map.
- **Long views** (Momentum & noise, Gap to GN, Edge & co-adaptation, Step rule & floors) open with a "Jump to" row of
  their titled sections.

## What changed (per view)

1. **Principles**: numbered cards with serif headlines and 17 px body text at about 80 characters per line; the cited views
   ("Step rule & floors: matched-loss chart", ...) are buttons that open that view.
2. **Curvature vs K-FAC**: one large panel per optimizer (2 x 2), training steps as a ramp of the optimizer's color with a
   step legend, "1 = K-FAC exact" labeled.
3. **Over training**: five panels, optimizer colors, "1 = K-FAC exact" on the mean-direction panel; added a labeled
   "our batch (2048 seq.)" line to the noise-scale panel (full width).
4. **Gap map**: six panels; the caveat is a callout; "our batch (2048 seq.)" on the b* panel, a 0 line on the alignment
   panel, and an added vertical "our batch" line on the reachable-share panel.
5. **Spectra**: one row per optimizer with the C and B spectra side by side; "1 = the mean eigenvalue" reference added.
6. **Exact Gauss-Newton**: saved states as large buttons; per input, the share-of-exact-GN chart as horizontal bars colored
   by direction family (hatched = shape of one direction with the per-matrix norms of the other) with value labels, next
   to the "which kind carries the gap" chart.
7. **Persistence**: Muon and PD side by side; curvature classes as an ordered ramp (stiffest darkest), filled/open markers
   for odd/even lags with a marker key.
8. **Coupling**: full-width grouped bars on a log axis, parts as a neutral ordered ramp and the whole update hatched; table
   with readable state names.
9. **Blocks & stability**: block-diagonal GN chart and staged chart full width with wrapped labels (staged inputs ordered by
   momentum weight); both stability tables with readable run names, raw keys, and ratios at or above 1 in bold.
10. **Momentum & noise**: the midpoint chart (about 24 runs in one chart) is split into one chart per batch and step; per
    transport state: key numbers as tiles, the staleness table, and the four binned curve charts with "all kinds pooled"
    in ink, one matrix kind highlighted in the optimizer's color and the other kinds faint (new "Highlighted matrix kind"
    control; the original colored all six kinds at once); eigenvector projections and one-step bars full width.
11. **Gap to GN**: anneal curves split by batch size (start step as a ramp); the one-step comparison with damped GN is drawn
    as a dot plot, one row per state and input (the original's 20 groups of bars needed vertical labels); the trainer chart
    is split by starting state (GN trainer from 100; Newton harness from Muon's and PD's 1M states at 500; 4M states at 183)
    with the baselines limited to each window.
12. **Edge & co-adaptation**: ten titled sections with a jump row (51 charts as of 2026-09-29): own step at own state;
    curvature bands (with the review 13:50 caution); momentum phase; secant multiplier by batch; the multiplier over
    training (1M); two-sided secant maps; cross-matrix coupling; layer-staged PD; coordination variants; momentum noise and
    gradient autocorrelation; interventions; pre-flight. Optimizer color with batch as line style and marker, as a two-part
    legend; labeled 1 and ½ lines on c* and a labeled −1 line on the flip chart (both from the original subtitles). See the
    2026-09-29 update log entry for how each new chart was laid out.
13. **Step rule & floors**: seven titled sections in the original order; the dual-axis normalized-GN chart is split into
    two single-axis charts; the matched-loss chart (19 to 23 series in one overlay) is split into 1M, 4M and 16M charts that
    share both axes; floor steps split by step rule, with the κ 2 arms hidden at first as in the original.
14. **Batch size**: the four 1M trajectories in one chart, then one chart per optimizer for the 4M runs, each with that
    optimizer's 1M trajectory as a thick reference, learning rate as a light-to-dark ramp and variants as line styles;
    finals table.

## Limits

- If a view's data are missing from the payload, the view or panel shows a "no data yet" note instead of an empty plot.
- A view added to the original Atlas later appears in the navigation with its explanation and a note that it has no
  readable layout yet.
- The "What to look for" captions and the per-panel subtitles are this edition's text; the view explanations and the
  Principles come from the original template at build time.

## Update log

- **2026-09-28 16:10 CDT: synced with the original Atlas as of 15:52.**
  - **Rebuilt payload:** byte-identical to the original builder's. The new data is all in Edge & co-adaptation: `edge.bands` 37 → 43 rows, `edge.interventions` 11 → 18 rows, and a new `edge.phase`.
  - **Principles:** the text is re-extracted, so the page follows the original's new wording.
  - **New Edge charts:** ported the original's new chart ("the fresher momentum is a phase effect: β 0.8 minus β 0.9 along training") as two panels.
    - The 16M-token runs (1× and 2× horizon) and the 4M-token runs are split. The original plots them on one step axis titled "16M tokens each", but the 4M runs count 4M-token steps.
    - Line styles follow the original: dashed = 2× horizon; dotted = second seed, clipped, or α¼; dash-dot and thicker = warmup.
  - **Verification:** `check_keys.mjs` all present; headless render of Edge and Principles at 1600 and 390 px with 0 console errors, 0 overflow and 0 missing-key reads (`build/screens_update/`).
  - **Republished** to https://claude.ai/artifact/1r6g3Sw7GVR8fZxxy4YzcN (Version 2).

- **2026-09-29 12:48 CDT: ported the original's new Edge & co-adaptation charts (template as of 11:16, data as of 12:43).**
  - **Rebuilt payload** (2.48 MB): byte-identical to the original builder's. New or changed since the last sync, all in `edge`:
    `phase` (17 series), new `secant` (`states`, `timelapse`, `twosided`), new `coupling` (19 states), new `staged`, new
    `coordination` (16M, 16M 2× horizon, 4M), new `momentum` (15 rows), and `interventions` 18 → 46. Between 12:20 and 12:43
    the in-progress "8 largest input directions" coordination arm grew from 13 to 57 lead points.
  - **Edge view reorganized** into ten titled sections with a jump row, in the original's order. Existing charts are kept;
    the band charts now carry the original's review caution (13:50: the lowest band is a single Lanczos node, so its c* > 1
    is not a band property), and the slope-share chart is retitled to "the lowest Lanczos node" as in the original.
  - **Momentum phase:** the 17 series are split into 16M 1× horizon (8), 16M 2× horizon (4) and 4M (5) on a shared y-axis;
    the 4M panel notes that Muon's pair is β 0.81 − β 0.9. Line styles: dotted = second seed or clip 0.1, thick dash-dot =
    β warmup, dashed = PD-top, long-dashed = PD α¼.
  - **Secant multiplier:** the 11-state chart is split by batch size on shared axes (states of one optimizer as a
    light-to-dark ramp; overall c* in the legend); the 1M time-lapse is one chart per optimizer on shared axes; labeled ½
    (stationary balance, review 03:46) and 1 (secant-optimal) lines. Bins with c* ≤ 0, which a log axis cannot show (the
    original draws them as lines dropping off the chart), are marked ▼ at the bottom edge with the value in the hover:
    one bin for PD 16M @9, and at 1M Muon @10 (3), SOAP-Muon @10 (2) and @50 (6), S∘PD @10 (1).
  - **Two-sided secant maps** (the original's six states): square heatmaps of log2(c*/½) with a color bar labeled in c*
    (⅛ … 2), masked below 0.2% of the step's energy as in the original, bin labels as powers of ten; the hover gives c*,
    the share of the step and of the first-order gain.
  - **Coupling:** the stacked chart with coherence on a second y-axis is split into two charts with the same row order: the
    step's GN curvature by pair of matrices (stacked to 100%, darker = closer to the matrix) and coherence per state, with a
    labeled "1 = uncorrelated pieces" line and the 0.8–2 in-sample band of exact GN's own pieces (review 03:46). The six
    48 × 48 correlation maps have per-layer ticks (L1 … L8), layer separators and kind order (q, k, v, o, up, down) in
    the subtitle; the hover names both matrices.
  - **Layer-staged PD:** difference to the control (training loss as lines, validation as markers, one legend entry per
    branch) and sharpness on a log axis with the control in ink.
  - **Coordination variants:** the 16M group's eight arms are split into full-correction variants (solid) and single-band
    variants (dashed), lead and sharpness charts each on shared axes; the 16M 2× horizon and 4M groups get their own pair of
    charts. Legend entries keep the original's final gap ("−0.152 at 92") or "(to step N)".
  - **Momentum noise:** the white-noise estimate chart (lines = all inputs, open markers = tail mean; PD 16M β 0.8 is a
    lighter shade, where the original drew it identical to β 0.9) and the autocorrelation chart split by batch size on
    shared axes; both review 21:48 caveats are in the section text, and the y-axis says "white-noise estimate ÷ |M|²"
    rather than "noise share", following the review.
  - **Interventions:** the 46 bars are split by batch size and horizon (16M 1× 25, 16M 2× 7, 4M 7, 1M 7), in the original
    order, each chart on its own x-range; labels break before "(vs …)" and never split "β 0.9"; bars keep the base
    optimizer's color, as before.
  - **Band and slope-share charts:** β 0.8 variants (new in `edge.bands`) get a lighter shade and open markers; before,
    "PD 16M" and "PD β0.8 16M" were identical lines in the slope-share chart.
  - **Principles:** 13 principles re-extracted; the new principle 8 cites "(Interventions)", which now links to the Edge view.
  - **Verification:** `node --check` of the page script; `check_keys.mjs` all present (all ten `edge` sub-keys); headless
    Chromium render of all 14 views at 1600 and 390 px and in dark mode with 0 console errors, 0 horizontal overflow, no
    empty panels and every chart drawn (Edge: 51/51 charts, 0 missing-key reads); screenshots of the Edge view reviewed by eye
    at 1600 px, 390 px and dark (`build/screens/`, `build/screens_edge0929/`).
  - **Republished** to https://claude.ai/artifact/1r6g3Sw7GVR8fZxxy4YzcN (Version 3).
