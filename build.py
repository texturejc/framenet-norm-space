"""Build frames.json: FrameNet 1.7 frames scored on 17 psycholinguistic norms.

Each frame's trigger words (lexical units) are matched to word norms.
A word gets a human rating where one exists (Warriner/Glasgow/Lancaster);
otherwise it gets the model estimate (mpnet + Ridge, English).
Every value is rescaled from its native scale to 0-1 (clipped).
Per frame and norm we store the mean, SD and number of matched words.

Usage: python build.py [--fn PATH_TO_FRAMENET_V17] [--norms PATH_TO_NORM_PROJECT]
"""
import argparse
import html
import json
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_FN = Path.home() / "Desktop/FrameNet/archive (1)/framenet_v17/framenet_v17"
DEFAULT_NORMS = HERE.parent / "norm_updated_extrapolations"
DEFAULT_CLUSTERS = HERE.parent / "Depression_anxiety/framenet_clusters_reduced.csv"
NS = "{http://framenet.icsi.berkeley.edu}"

NORMS = [
    "valence", "arousal", "dominance",
    "concreteness", "imageability", "semantic_size",
    "auditory", "gustatory", "haptic", "interoceptive", "olfactory", "visual",
    "foot_leg", "hand_arm", "head", "mouth", "torso",
]
# Native rating scales; every value is rescaled to 0-1 with these bounds
SCALES = {n: (1, 9) for n in ["valence", "arousal", "dominance"]}
SCALES.update({n: (1, 7) for n in ["concreteness", "imageability", "semantic_size"]})
SCALES.update({n: (0, 5) for n in NORMS if n not in SCALES})


def unit(v, norm):
    """Native rating -> 0-1, clipped (a few model estimates fall just outside the scale)."""
    lo, hi = SCALES[norm]
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


def clean_lemma(name: str) -> str:
    """'(can't) help' -> 'help'; lowercase; collapse spaces."""
    s = re.sub(r"\([^)]*\)", " ", name)
    return re.sub(r"\s+", " ", s).strip().lower()


def frame_definition(path: Path, max_len: int = 420) -> str:
    """Frame definition text before the first example, tags stripped."""
    raw = ET.parse(path).getroot().find(f"{NS}definition").text or ""
    raw = html.unescape(raw).split("<ex>")[0]
    text = re.sub(r"<[^>]+>", "", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len].rsplit(" ", 1)[0] + "…"


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", type=Path, default=DEFAULT_FN)
    ap.add_argument("--norms", type=Path, default=DEFAULT_NORMS)
    ap.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    ap.add_argument("--out", type=Path, default=HERE / "frames.json")
    a = ap.parse_args()

    # --- norms: human ratings, then English model estimates as fallback
    gold = pd.read_parquet(a.norms / "data/train_norms.parquet")[NORMS]
    gold.index = gold.index.str.lower()
    gold = gold[~gold.index.duplicated()]
    est = pd.read_parquet(a.norms / "outputs/all_norm_estimates_multilingual.parquet")
    est = est.loc[est.language == "en", NORMS]
    est.index = est.index.str.lower()
    est = est[~est.index.duplicated()]

    def lookup(lemma):
        """Return (values, is_human_flags) per norm, or None if no match."""
        g = gold.loc[lemma] if lemma in gold.index else None
        e = est.loc[lemma] if lemma in est.index else None
        if g is None and e is None:
            return None
        vals, human = [], []
        for n in NORMS:
            gv = None if g is None or pd.isna(g[n]) else float(g[n])
            if gv is not None:
                vals.append(unit(gv, n)); human.append(1)
            elif e is not None and not pd.isna(e[n]):
                vals.append(unit(float(e[n]), n)); human.append(0)
            else:
                vals.append(np.nan); human.append(0)
        return vals, human

    # --- FrameNet
    frames = [(f.get("name"), f.get("ID"))
              for f in ET.parse(a.fn / "frameIndex.xml").getroot().findall(f"{NS}frame")]
    lus_by_frame = defaultdict(list)
    for lu in ET.parse(a.fn / "luIndex.xml").getroot().findall(f"{NS}lu"):
        lus_by_frame[lu.get("frameName")].append(lu.get("name"))

    clusters = pd.read_csv(a.clusters).set_index("frame")["cluster_label"].to_dict()

    out_frames = []
    n_lu = n_hit = 0
    for name, fid in frames:
        lus = sorted(lus_by_frame.get(name, []))
        if not lus:
            continue  # non-lexical frame: nothing to score
        words = {}  # lemma -> [lu names], dedupe lemmas across POS
        for lu in lus:
            lemma, pos = lu.rsplit(".", 1)
            words.setdefault(clean_lemma(lemma), []).append(lu)
        matched, unmatched = [], []
        for lemma, lu_names in words.items():
            n_lu += len(lu_names)
            res = lookup(lemma)
            if res is None:
                unmatched.append(" / ".join(lu_names))
                continue
            n_hit += len(lu_names)
            vals, human = res
            # h: 1 = all values human-rated, 0 = all estimated, else a per-norm list
            h = 1 if all(human) else 0 if not any(human) else human
            matched.append({"w": " / ".join(lu_names), "v": [r(v) for v in vals], "h": h})
        mean = sd = [None] * len(NORMS)
        k = [0] * len(NORMS)  # words with a value, per norm
        if matched:
            M = np.array([[np.nan if x is None else x for x in m["v"]] for m in matched], float)
            k = (~np.isnan(M)).sum(0)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = [r(x) for x in np.nanmean(M, 0)]
                sd = [r(s) if c > 1 else None for s, c in zip(np.nanstd(M, 0, ddof=1), k)]
            k = [int(c) for c in k]
        out_frames.append({
            "name": name, "id": int(fid),
            "group": clusters.get(name, "Unassigned"),
            "def": frame_definition(a.fn / "frame" / f"{name}.xml"),
            "n": len(matched), "n_lu": len(lus),
            "mean": mean, "sd": sd, "k": k,
            "words": matched, "unmatched": unmatched,
        })

    payload = {
        "norms": NORMS,
        "native_scales": {n: list(SCALES[n]) for n in NORMS},
        "meta": {
            "framenet": "FrameNet 1.7",
            "frames_scored": sum(1 for f in out_frames if f["n"] > 0),
            "frames_lexical": len(out_frames),
            "lu_total": n_lu, "lu_matched": n_hit,
        },
        "frames": out_frames,
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    a.out.write_text(blob)
    # Same data as a script, so index.html also works when opened straight from disk (file://)
    a.out.with_suffix(".js").write_text("window.FRAME_DATA=" + blob + ";\n")
    print(f"wrote {a.out} ({a.out.stat().st_size/1e3:.0f} KB): "
          f"{payload['meta']['frames_scored']}/{len(out_frames)} frames scored, "
          f"{n_hit}/{n_lu} LUs matched")


if __name__ == "__main__":
    main()
