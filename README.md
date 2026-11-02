# FrameNet Norm Space

**Live:** https://texturejc.github.io/framenet-norm-space/

Interactive 3D map of the ~1,060 FrameNet 1.7 semantic frames that have trigger words with
norm values. Each frame is placed by the average psycholinguistic norm ratings of its trigger
words (lexical units). Any of 17 norms can be chosen for each axis: affective, conceptual,
perceptual or action-effector.

## Views
- **3D map**: any three of the 17 norms as axes; colour by frame group or by a norm.
- **Rankings**: frames ranked from lowest to highest on a summary index or any norm, with 95% CIs.
  - *Sensorimotor intensity*: Minkowski-3 strength over the 11 Lancaster dimensions.
  - *Affective intensity*: mean of valence extremity (|valence − 0.5| × 2) and arousal.
  - *Concreteness*: Glasgow concreteness.
  - Each index is computed per word, then averaged per frame. Rankings default to frames with at least 3 words.

## Files
- `index.html`: the viewer (Plotly.js, no build step; loads `frames.js`, so it also works opened straight from disk).
- `frames.js` / `frames.json`: identical data (script / plain JSON); per-frame mean / SD / n for each norm, plus per-word values.
- `build.py`: regenerates `frames.json` from FrameNet 1.7 and the norm project.

## Data
- **FrameNet 1.7**: `luIndex.xml`, `frameIndex.xml`, `frame/*.xml` (definitions).
- **Norms** (from `../norm_updated_extrapolations`):
  - Human ratings: Warriner et al. (2013) and Glasgow Norms (VAD averaged where both exist);
    Glasgow (concreteness, imageability, semantic size); Lancaster Sensorimotor Norms (11 dimensions).
  - Fallback: model estimates (Ridge on multilingual mpnet embeddings; LaBSE for concreteness,
    imageability and gustatory; cross-validated r = .60–.90) for words without a human rating. The viewer shows estimated values in grey italic.
- **Frame groups**: 46 Louvain communities on FrameNet's frame-relation graph
  (`../Depression_anxiety/framenet_clusters_reduced.csv`).

## Method notes
- Lemmas are lowercased; bracketed parts are removed (`(can't) help` → `help`); a lemma
  listed under several parts of speech in one frame counts once.
- 11,963 / 13,572 trigger words match. 148 frames have no trigger words and are excluded.
  12 more frames have trigger words but none match a norm.
- All values are rescaled from their native scale (1–9, 1–7 or 0–5) to 0–1. About 120 model estimates
  that fell slightly outside their scale are clipped to 0 or 1 before averaging.
- SD is left blank for frames with fewer than 2 matched words on that norm.

## Rebuild
    python build.py --fn "/path/to/framenet_v17" --norms ../norm_updated_extrapolations
