# AI Usage Log — EEG Explainer (model + assets)

Per the course AI policy, this discloses where AI (Claude, via Claude Code) was
used on the model/asset code, what it was asked to do, and what it produced.

## What AI was asked to do

1. **Review + package the existing code into a standalone repo.** The core
   trainer/exporter (`train_and_export.py`, `EEGExplainerNet` + MNE data path +
   `/assets` exporter) was authored for the project; AI was asked to organize it
   into a clean, runnable repository with README, dependency files, a one-shot
   data-fetch helper (`download_data.py`), and `.gitignore`.
2. **Verify the pipeline.** AI ran the built-in `--smoke` path (synthetic data,
   no download) end to end and inspected the exported `meta.json` / `model.json`
   / `trials/*.json` to confirm shapes and self-consistency.
3. **One correctness fix (flagged, not silent).** The smoke run reproduced a
   `SIGBUS` crash on the auto-selected **MPS** device under torch 2.0.0. AI added
   an additive `--device {auto,cpu,mps,cuda}` flag (default `auto` = prior
   behavior) so the crash is escapable, and documented the torch≥2.1 fix.

## Verification performed

- `--smoke --device cpu` runs clean end to end (exit 0).
- Exported arrays are self-consistent: `logits == pooled·dense_W + b`,
  max reconstruction error `0.00e+00`.
- Asset bundle ~1.2 MB (< 2 MB target); all six stage tensors present per trial;
  curated picks (clean L/R, confidently-wrong, low-conf, artifact) populated.
- Stage shapes confirmed: `spatial_W [8×64]`, `temporal_K [8×25]`,
  `dense_W [304×2]`; per-trial stages `[8×640]`.

## Environment note

Verified on macOS / Apple Silicon, Python 3.10, torch 2.0.0, mne 1.11, numpy
1.26.4. `environment.yml` pins torch ≥ 2.1 (recommended); on torch 2.0.x use
`--device cpu` to avoid the MPS SIGBUS. Real training pulls PhysioNet EEGMMIDB
subject data via MNE on demand.
