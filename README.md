# EEG Explainer — model + assets

**CSE 151B project · Communication track · Brendan Keane**

A small, teachable **EEGNet-inspired CNN** that decides *imagined left fist* vs
*imagined right fist* from raw scalp EEG, plus the exporter that dumps every
intermediate stage as static JSON. Those assets are what the browser explainer
scrubs through, stage by stage — raw trial → montage → spatial filter → temporal
filter → energy (band power) → decision — so a newcomer can *watch* a convolution
read a brain signal instead of meeting a wall of equations.

This repo is the **code, data handling, and model** half of the project (the
front-end UI is a separate deliverable). Inference is pre-computed: the network
trains once, and the UI never recomputes anything.

---

## How to run

### 1. Environment
```bash
conda env create -f environment.yml     # creates the `eeg-explainer` env
conda activate eeg-explainer
# — or, in any Python 3.10 env —
pip install -r requirements.txt
```
Device is auto-selected (cuda → mps → cpu). **On Apple Silicon with torch 2.0.x,
prefer `--device cpu`** — that torch's MPS backend can `SIGBUS` on this model's
backward pass (fixed in torch ≥ 2.1). Override anytime with `--device {cpu,mps,cuda,auto}`.

### 2. Data (optional pre-fetch)
```bash
python download_data.py --subject 1
```
`train_and_export.py` downloads on demand, so this is only to warm the cache.
Data: **EEG Motor Movement/Imagery Database (EEGMMIDB)**, PhysioNet v1.0.0 —
109 subjects · 64 channels · 160 Hz. Runs **4, 8, 12** are the imagined
open/close **LEFT (T1)** vs **RIGHT (T2)** fist task. MNE caches under
`~/mne_data/MNE-eegbci-data/.../eegmmidb/1.0.0/`. License: ODC-By 1.0 (cite on use).

### 3. Train + export assets
```bash
# smoke test — synthetic data, no download, verifies the whole pipeline:
python train_and_export.py --smoke --device cpu --out ./assets_smoke

# real, within-subject (subject 1):
python train_and_export.py --subject 1 --epochs 200 --out ./assets
```
Within-subject accuracy typically lands ~70–85% (a defensible per-trial number;
accuracy is *not* the goal — see Design choices). Each run prints the split
balance, best val / test accuracy, the logit-reconstruction error (should be
`~0`), total asset size, and the auto-nominated curated trials.

---

## What gets exported (`--out` directory)

The exporter writes exactly what the UI reads — nothing is recomputed downstream.
Verified self-consistent: `logits == pooled · dense_W + b` to machine precision.

| File | Contents |
|---|---|
| `meta.json` | `sfreq`, window, `n_components`, kernel/pool params, `display_channels` (8), 2-D montage coords, temporal-response freqs, and the **curated** trial picks |
| `model.json` | learned weights: `spatial_W [8×64]`, `temporal_K [8×25]`, `temporal_response [8×64]` (\|FFT\| of each kernel), `dense_W [n_pooled×2]`, `dense_b [2]` |
| `trials/trial_NN.json` | one curated trial: `label`/`pred`/`correct`/`confidence`/`split`, then every stage — `raw_display [8×640]`, `spatial_out [8×640]`, `temporal_out [8×640]`, `squared [8×640]`, `pooled [n_pooled]`, `contributions [n_pooled×2]`, `logits`, `probs` |

Only the curated set is written per-trial (size discipline — bundle stays < ~2 MB).
`curated` groups them as `clean_left`/`clean_right` (2 each), `confidently_wrong`,
`low_confidence`, `artifact` (frontal-blink), `free_explore`. Edit `meta.json` to
override the picks by hand.

## Model / stage map

Kept identical to the design spec so the UI narration lines up 1:1:

| Stage | Layer | Meaning |
|---|---|---|
| 2 spatial | `Conv2d(1, 8, (64,1))` | mix 64 electrodes → 8 virtual channels |
| 3 temporal | `Conv2d(8, 8, (1,25), groups=8)` | per-channel learned bandpass (BatchNorm folded into `temporal_out` so `squared == temporal_out²` holds exactly) |
| 4 energy | `square → AvgPool(1,75)/15 → log` | oscillation amplitude → band power |
| 5 decision | `Linear(n_pooled → 2)` → softmax | logits → L/R probability |

This deliberately **reorders EEGNet's temporal-then-spatial design to spatial-then-temporal**
— a legibility tradeoff (each stage stays readable), not a faithful EEGNet
reimplementation, hence "EEGNet-*inspired*."

## Design choices

- **Within-subject** is the honest framing for a per-trial explainer and the
  easiest path to a defensible number. Every trial records which split
  (`train`/`val`/`test`) it came from so the write-up stays honest.
- **No leakage:** per-channel z-score is fit on **training** trials only, then
  applied to val/test. Split is **stratified by label** (matters a lot for the
  ~45-trial within-subject sets).
- **Accuracy is not the goal.** One *confidently-wrong* trial is the most
  valuable teaching asset in the project; the exporter auto-nominates candidates.

## Assumptions / notes

1. **Epoch window** is trimmed to exactly `TARGET_T = 640` samples (4 s @ 160 Hz)
   so every stage tensor has a fixed length the UI can rely on.
2. **Display channels** are the 8 motor-relevant electrodes
   (`C3 C4 Cz FC3 FC4 CP3 CP4 Fpz`); if a named channel is absent it's padded by
   index so the export never fails.
3. **Blink/artifact** trials are picked by frontal peak-to-peak amplitude
   (`Fp1 Fpz Fp2 AF7 AF8`) — a heuristic, not a trained artifact detector.

AI use for this repo is disclosed in [`AI_USAGE_LOG.md`](AI_USAGE_LOG.md);
running design decisions are in [`DECISIONS.md`](DECISIONS.md).

## Repo layout
```
train_and_export.py   # model (EEGExplainerNet) + data loading + training + asset export
download_data.py      # optional one-shot PhysioNet pre-fetch
requirements.txt      # pip deps         environment.yml  # conda env
assets/               # generated bundle the UI consumes (git-ignored)
```
