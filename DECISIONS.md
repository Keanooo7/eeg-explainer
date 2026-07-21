# DECISIONS.md — EEG Explainer

Running log of design decisions: what I chose, why, and what it cost. Kept since
day one so the Week 5 experience report works from real notes, not a
reconstruction. (This file seeds the model/asset decisions; append as the project
moves.)

## Model / task framing

- **Within-subject, binary L/R imagined fist.** Chose the honest, defensible
  per-trial framing over cross-subject (which needs far more data and muddies a
  per-trial explainer). Cost: accuracy numbers don't generalize across subjects —
  stated plainly in the write-up. The goal is understanding, not a leaderboard.
- **Spatial-then-temporal filter order.** Deliberate reorder of EEGNet's
  temporal-then-spatial design, named as a *legibility* tradeoff so each stage
  stays readable. Cost: no longer a faithful EEGNet reimplementation — hence
  "EEGNet-**inspired**," not "EEGNet-style."
- **Explicit energy stage (square → avg-pool → log = band power).** Named the step
  the original abstract skipped between the temporal convolution and the decision,
  because "a wiggle becomes a single number" is the load-bearing intuition of the
  whole explainer.

## Correctness / honesty guards (in `train_and_export.py`)

- **No test leakage:** per-channel z-score fit on training trials only.
- **Stratified 60/20/20 split** by label, because the ~45-trial within-subject
  sets otherwise hand you a lopsided val/test.
- **BatchNorm folded into `temporal_out`** so the exported `squared` equals
  `temporal_out²` exactly — the UI can trust the arrays instead of re-deriving.

## Tooling

- **Added `--device {auto,cpu,mps,cuda}`** (2026-07-20). Auto-MPS `SIGBUS`-crashed
  on torch 2.0.0 during the backward pass; the flag lets me force CPU. Default
  stays `auto`, so nothing changes on torch ≥ 2.1. Cost: one extra CLI arg.

## Scope cuts (if I fall behind, in this order)

Story-lens interactivity → narration text only; draggable stage-3 kernel →
static figure; eight curated trials → five. **Never cut:** the confidently-wrong
trial, the energy stage, the metaphor-leak panels.
