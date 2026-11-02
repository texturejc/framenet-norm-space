# FrameNet Norm Space

Interactive 3D map of the ~1,060 FrameNet 1.7 semantic frames that have trigger words with
norm values. Each frame is placed by the average psycholinguistic norm ratings of its trigger
words (lexical units). Any of 17 norms can be chosen for each axis: affective, conceptual,
perceptual or action-effector.

## Files
- `index.html`: the viewer (Plotly.js, no build step; loads `frames.js`, so it also works opened straight from disk).
- `frames.js` / `frames.json`: identical data (script / plain JSON); per-frame mean / SD / n for each norm, plus per-word values.
- `build.py`: regenerates `frames.json` from FrameNet 1.7 and the norm project.

## Data
- **FrameNet 1.7**: `luIndex.xml`, `frameIndex.xml`, `frame/*.xml` (definitions).
- **Norms** (from `../norm_updated_extrapolations`):
  - Human ratings: Warriner et al. (2013) and Glasgow Norms (VAD averaged where both exist);
    Glasgow (concreteness, imageability, semantic size); Lancaster Sensorimotor Norms (11 dimensions).
  - Fallback: model estimates (Ridge on multilingual mpnet embeddings, English) for words
    without a human rating. The viewer shows estimated values in grey italic.
- **Frame groups**: 46 Louvain communities on FrameNet's frame-relation graph
  (`../Depression_anxiety/framenet_clusters_reduced.csv`).

## Method notes
- Lemmas are lowercased; bracketed parts are removed (`(can't) help` → `help`); a lemma
  listed under several parts of speech in one frame counts once.
- 11,963 / 13,572 trigger words match. 148 frames have no trigger words and are excluded.
  12 more frames have trigger words but none match a norm.
- SD is left blank for frames with fewer than 2 matched words on that norm.

## Rebuild
    python build.py --fn "/path/to/framenet_v17" --norms ../norm_updated_extrapolations
