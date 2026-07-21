#!/usr/bin/env python3
"""
EEG explainer — train + asset export
====================================
CSE 151B communication track · Brendan Keane

Trains the small, teachable EEGNet-inspired CNN from the design spec (§3) on the
PhysioNet EEG Motor Movement/Imagery Database (imagined LEFT vs RIGHT fist, runs
4/8/12, within-subject) and dumps the exact `/assets` bundle the explainer UI
expects (§4). Every stage of the model has a corresponding array in the export
so the front-end never has to recompute anything.

Dataset (exact)
---------------
* Name    : EEG Motor Movement/Imagery Dataset (EEGMMIDB), PhysioNet, v1.0.0
* Released: 2009-09-09   ·   DOI: 10.13026/C28G6P
* License : Open Data Commons Attribution License v1.0 (ODC-By 1.0) — cite on use
* Content : 109 subjects · 64 EEG channels · 160 Hz · BCI2000 acquisition
* Runs used: 4, 8, 12  = "imagine opening and closing the LEFT or RIGHT fist"
             (task 2 of the protocol; the three imagined-fist repetitions)
* Labels  : per-run annotations  T0 = rest, T1 = left fist, T2 = right fist
* Access  : pulled automatically by mne.datasets.eegbci.load_data(subject, runs),
            cached under ~/mne_data/MNE-eegbci-data/.../eegmmidb/1.0.0/

Pipeline / stage map (kept identical to the spec so the UI narration lines up):
    STAGE 2  spatial filter    Conv2d(1, 8, (64,1))      -> 8 virtual channels
    STAGE 3  temporal filter   Conv2d(8, 8, (1,25), g=8) -> learned bandpass
             (BatchNorm folded into the exported temporal_out so that
              squared == temporal_out**2 holds exactly for the visualization)
    STAGE 4  energy            square -> avg-pool(1,75)/15 -> log = band power
    STAGE 5  decision          Linear(n_pooled -> 2) -> logits -> softmax

Run for real:
    pip install torch mne numpy
    python train_and_export.py --subject 1 --epochs 200 --out ./assets

Verify the plumbing without any data download (synthetic tensors, full export):
    python train_and_export.py --smoke --out ./assets_smoke

Revisions vs the first draft (all three fix correctness / honesty, not behavior of the UI)
-----------------------------------------------------------------------------------------
* [BUG]     load_real_data built the MNE event_id as {"T1":"T1", ...}; MNE requires
            integer codes, so the real-data path raised TypeError before training.
            Now maps description -> integer code: {k: event_id[k] for k ...}.
* [LEAK]    per-channel z-score is now fit on TRAINING trials only and applied to
            val/test, so the reported accuracy isn't inflated by test statistics.
* [SPLIT]   60/20/20 split is now stratified by label, so tiny within-subject sets
            don't hand you a lopsided val/test (which would make val-based model
            selection and the reported test_acc noisy).

Notes
-----
* Within-subject is the honest framing for a per-trial explainer and the easiest
  path to a defensible number (~70-85%). Curated trials are drawn from that one
  subject; each trial records which split it came from so the write-up can be honest.
* MPS (Apple Silicon) is auto-selected — good for the Mac Studio.
* Accuracy is NOT the goal. One confidently-wrong trial is the most valuable asset
  in the project (§8); the script auto-nominates candidates for you to curate.
"""

import argparse, json, os, math
import numpy as np

# torch is imported lazily inside main so `--help` works without it installed.

# ----------------------------------------------------------------------------- config
DISPLAY_CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "Fpz"]  # §4 raw_display (8 only)
FRONTAL_CHANNELS = ["Fp1", "Fpz", "Fp2", "AF7", "AF8"]                     # blink heuristic
N_COMPONENTS = 8       # spatial filters / virtual channels
KERNEL_LEN   = 25      # temporal kernel taps
POOL_LEN     = 75
POOL_STRIDE  = 15
WINDOW_S     = 4.0
TARGET_T     = 640     # 4 s @ 160 Hz — enforced by trimming epochs
ROUND        = 4       # decimals in the JSON (size discipline, §4)

# ----------------------------------------------------------------------------- model
def build_model_module():
    import torch
    import torch.nn as nn

    class EEGExplainerNet(nn.Module):
        """Spatial -> temporal -> square/pool/log -> linear. See stage map above."""
        def __init__(self, n_chan=64, n_comp=N_COMPONENTS, klen=KERNEL_LEN, t_len=TARGET_T):
            super().__init__()
            self.spatial  = nn.Conv2d(1, n_comp, kernel_size=(n_chan, 1), bias=False)          # STAGE 2
            self.temporal = nn.Conv2d(n_comp, n_comp, kernel_size=(1, klen),
                                      groups=n_comp, padding=(0, klen // 2), bias=False)        # STAGE 3
            self.bn       = nn.BatchNorm2d(n_comp)
            self.pool     = nn.AvgPool2d(kernel_size=(1, POOL_LEN), stride=(1, POOL_STRIDE))    # STAGE 4
            self.drop     = nn.Dropout(0.5)
            n_pooled      = n_comp * ((t_len - POOL_LEN) // POOL_STRIDE + 1)
            self.fc       = nn.Linear(n_pooled, 2)                                              # STAGE 5
            self.n_pooled = n_pooled

        def stages(self, x):
            """Return every intermediate the explainer needs (eval-mode, no dropout)."""
            import torch
            s  = self.spatial(x)                              # (B, C, 1, T)
            t  = self.temporal(s)                             # (B, C, 1, T)
            tb = self.bn(t)                                   # post-BN == exported temporal_out
            sq = tb ** 2                                      # STAGE 4a
            pl = self.pool(sq)                                # (B, C, 1, P)
            lg = torch.log(torch.clamp(pl, min=1e-6))        # STAGE 4c band power
            flat = lg.flatten(1)                              # (B, n_pooled)
            logits = self.fc(flat)                            # STAGE 5
            return dict(spatial=s, temporal=tb, squared=sq, pooled=lg, flat=flat, logits=logits)

        def forward(self, x):
            s  = self.spatial(x)
            t  = self.temporal(s)
            tb = self.bn(t)
            lg = torch.log(torch.clamp(self.pool(tb ** 2), min=1e-6))
            return self.fc(self.drop(lg.flatten(1)))

    return EEGExplainerNet

# ----------------------------------------------------------------------------- data
def azimuthal_2d(ch_pos_3d):
    """Project 3D montage coords to a top-down 2D topomap layout (nose up),
    normalized to roughly [-0.5, 0.5]. Matches the spec's example (C3 ~ x:-0.42, y:0)."""
    names = list(ch_pos_3d.keys())
    P = np.array([ch_pos_3d[n] for n in names], dtype=float)
    P = P - P.mean(0, keepdims=True)                      # center
    norm = np.linalg.norm(P, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = P / norm
    phi = np.arccos(np.clip(U[:, 2], -1, 1))             # polar angle from +z (vertex)
    az  = np.arctan2(U[:, 1], U[:, 0])                    # +x right, +y nose(front)
    X = phi * np.cos(az)
    Y = phi * np.sin(az)
    r = max(np.max(np.abs(X)), np.max(np.abs(Y))) or 1.0
    X = 0.5 * X / r
    Y = 0.5 * Y / r
    return names, {n: (float(X[i]), float(Y[i])) for i, n in enumerate(names)}

def load_real_data(subject, sfreq_target=160.0, l_freq=1.0, h_freq=40.0):
    """Load imagined L/R fist (runs 4,8,12) for one subject via MNE. Returns
    X (N,64,T) float32 *un-normalized*, y (N,), ch_names[64], pos2d dict, sfreq.

    NOTE: normalization is intentionally deferred to main() so the z-score can be
    fit on training trials only (see the [LEAK] revision note)."""
    import mne
    from mne.datasets import eegbci
    from mne import Epochs, pick_types, events_from_annotations

    runs = [4, 8, 12]  # imagined open/close LEFT or RIGHT fist (EEGMMIDB task 2)
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)                               # -> standard 10-05 names
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="warn")
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
    if abs(raw.info["sfreq"] - sfreq_target) > 1e-3:
        raw.resample(sfreq_target, verbose="ERROR")

    # For fist runs: T1 = left fist, T2 = right fist.
    events, event_id = events_from_annotations(raw, verbose="ERROR")
    want = {}
    for k, v in event_id.items():
        if k in ("T1", "left"):  want[v] = 0
        if k in ("T2", "right"): want[v] = 1

    picks = pick_types(raw.info, eeg=True, exclude="bads")
    # [BUG FIX] event_id values MUST be integer codes, not the description strings.
    sel_event_id = {k: event_id[k] for k in event_id if k in ("T1", "T2")}
    epochs = Epochs(raw, events, event_id=sel_event_id,
                    tmin=0.0, tmax=(TARGET_T) / sfreq_target, picks=picks,
                    baseline=None, preload=True, verbose="ERROR")
    X = epochs.get_data(copy=True).astype(np.float32)     # (N, 64, ~T)
    X = X[:, :, :TARGET_T]                                # trim to exactly 640
    codes = epochs.events[:, -1]
    y = np.array([want[c] for c in codes], dtype=np.int64)
    ch_names = epochs.ch_names

    m = raw.get_montage().get_positions()["ch_pos"]
    m = {n: m[n] for n in ch_names if n in m and not np.any(np.isnan(m[n]))}
    names_ordered, pos2d = azimuthal_2d(m)

    return X, y, ch_names, pos2d, float(sfreq_target)

def load_smoke_data(n=80, seed=0):
    """Synthetic data with a *learnable* left/right signal so the full pipeline
    (train -> export -> consistency checks) runs with no download. Returns
    un-normalized X so main() can fit the z-score on training trials only."""
    rng = np.random.default_rng(seed)
    ch_names = _fake_64_names()
    T = TARGET_T
    X = rng.standard_normal((n, 64, T)).astype(np.float32) * 0.5
    y = rng.integers(0, 2, size=n).astype(np.int64)
    t = np.arange(T) / 160.0
    mu = np.sin(2 * np.pi * 10 * t).astype(np.float32)    # 10 Hz mu rhythm
    c3, c4 = ch_names.index("C3"), ch_names.index("C4")
    for i in range(n):
        # imagined left  -> suppress mu over RIGHT (C4); right -> suppress over LEFT (C3)
        X[i, c3] += (0.2 if y[i] == 0 else 1.0) * mu
        X[i, c4] += (1.0 if y[i] == 0 else 0.2) * mu
    pos2d = {nm: (math.cos(k), math.sin(k)) for k, nm in enumerate(ch_names)}
    pos2d = {nm: (0.5 * x, 0.5 * z) for nm, (x, z) in pos2d.items()}
    return X, y, ch_names, pos2d, 160.0

def _fake_64_names():
    base = DISPLAY_CHANNELS + FRONTAL_CHANNELS
    base = list(dict.fromkeys(base))
    i = 0
    while len(base) < 64:
        base.append(f"E{i:02d}"); i += 1
    return base[:64]

def stratified_split(y, seed=0, fracs=(0.6, 0.2, 0.2)):
    """Per-class 60/20/20 split so val/test class balance tracks the data — matters
    a lot for the ~45-trial within-subject sets. Returns (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        nc = len(idx)
        n_tr = int(round(fracs[0] * nc))
        n_va = int(round(fracs[1] * nc))
        tr += idx[:n_tr].tolist()
        va += idx[n_tr:n_tr + n_va].tolist()
        te += idx[n_tr + n_va:].tolist()
    tr, va, te = np.array(tr, int), np.array(va, int), np.array(te, int)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

def zscore_fit_apply(X, train_idx, eps=1e-7):
    """Fit per-channel mean/std on TRAINING trials only, apply to every trial.
    Per-channel scaling is a constant across trials, so relative amplitudes
    (e.g. frontal peak-to-peak used for the blink pick) are preserved."""
    mu = X[train_idx].mean(axis=(0, 2), keepdims=True)
    sd = X[train_idx].std(axis=(0, 2), keepdims=True) + eps
    return ((X - mu) / sd).astype(np.float32)

# ----------------------------------------------------------------------------- train
def train(model, Xtr, ytr, Xva, yva, device, epochs=200, lr=1e-3, wd=1e-4, verbose=True):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr[:, None], device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva[:, None], device=device)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr_t)
        loss = lossf(out, ytr_t)
        loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = (model(Xva_t).argmax(1).cpu().numpy() == yva).mean() if len(yva) else 0.0
            if va >= best:
                best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  epoch {ep+1:3d}  loss {loss.item():.3f}  val_acc {va:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best

# ----------------------------------------------------------------------------- export
def r(a):
    return np.round(np.asarray(a, dtype=float), ROUND).tolist()

def temporal_response(kernels, sfreq, nbins=64):
    """|FFT| of each temporal kernel -> shows it's a bandpass. Returns ([8][nbins], freqs[nbins])."""
    n = (nbins - 1) * 2
    resp, freqs = [], np.fft.rfftfreq(n, d=1.0 / sfreq)[:nbins]
    for k in kernels:
        mag = np.abs(np.fft.rfft(k, n=n))[:nbins]
        mag = mag / (mag.max() + 1e-9)
        resp.append(mag)
    return np.array(resp), freqs

def pick_curated(probs, preds, y, blink_score):
    """Nominate 8 trials: 2 clean L, 2 clean R, 1 confidently-wrong, 1 low-conf, 1 blink, 1 free."""
    conf = probs.max(1)
    correct = preds == y
    order_conf = np.argsort(-conf)
    picks, used = {"clean_left": [], "clean_right": [], "confidently_wrong": [],
                   "low_confidence": [], "artifact": []}, set()
    def take(cat, idx):
        if idx not in used:
            picks[cat].append(int(idx)); used.add(int(idx))
    for i in order_conf:                                  # clean, high-confidence, correct
        if len(picks["clean_left"]) < 2 and correct[i] and y[i] == 0: take("clean_left", i)
        if len(picks["clean_right"]) < 2 and correct[i] and y[i] == 1: take("clean_right", i)
    for i in order_conf:                                  # confidently wrong
        if len(picks["confidently_wrong"]) < 1 and not correct[i]: take("confidently_wrong", i)
    for i in np.argsort(np.abs(conf - 0.55)):             # ~55% confidence
        if len(picks["low_confidence"]) < 1 and i not in used: take("low_confidence", i); break
    for i in np.argsort(-blink_score):                    # strongest frontal p-p
        if len(picks["artifact"]) < 1 and i not in used: take("artifact", i); break
    free = None
    for i in order_conf:
        if i not in used: free = int(i); break
    return picks, free

def export(model, X, y, ch_names, pos2d, sfreq, out_dir, device, split_of=None):
    import torch
    os.makedirs(os.path.join(out_dir, "trials"), exist_ok=True)
    split_of = split_of or {}
    model.eval()
    Xt = torch.tensor(X[:, None], device=device)
    with torch.no_grad():
        st = model.stages(Xt)
    logits = st["logits"].cpu().numpy()
    probs  = _softmax(logits)
    preds  = probs.argmax(1)

    # weights
    spatial_W = model.spatial.weight.detach().cpu().numpy().reshape(N_COMPONENTS, -1)   # [8][64]
    temporal_K = model.temporal.weight.detach().cpu().numpy().reshape(N_COMPONENTS, -1)  # [8][25]
    fcW = model.fc.weight.detach().cpu().numpy()   # (2, n_pooled)
    fcB = model.fc.bias.detach().cpu().numpy()     # (2,)
    dense_W = fcW.T                                 # [n_pooled][2]
    tresp, tfreqs = temporal_response(temporal_K, sfreq)

    # display-channel indices (robust, case-insensitive) + frontal blink score
    lower = {c.lower(): i for i, c in enumerate(ch_names)}
    disp_idx = [lower[c.lower()] for c in DISPLAY_CHANNELS if c.lower() in lower]
    if len(disp_idx) < 8:                           # pad if a named channel is absent
        disp_idx += [i for i in range(len(ch_names)) if i not in disp_idx][:8 - len(disp_idx)]
    disp_idx = disp_idx[:8]
    front_idx = [lower[c.lower()] for c in FRONTAL_CHANNELS if c.lower() in lower] or [0]
    blink_score = np.ptp(X[:, front_idx, :], axis=2).mean(1)   # per-trial frontal peak-to-peak

    picks, free = pick_curated(probs, preds, y, blink_score)
    curated = {k: [f"trial_{i:02d}" for i in v] for k, v in picks.items()}
    chosen = [i for v in picks.values() for i in v] + ([free] if free is not None else [])
    curated["free_explore"] = [f"trial_{free:02d}"] if free is not None else []

    # ---- meta.json
    meta = {
        "sfreq": sfreq, "window_s": WINDOW_S, "n_components": N_COMPONENTS,
        "kernel_len": KERNEL_LEN, "target_T": TARGET_T,
        "pool": {"len": POOL_LEN, "stride": POOL_STRIDE},
        "display_channels": [ch_names[i] for i in disp_idx],
        "montage": [{"name": n, "x": round(pos2d[n][0], 4), "y": round(pos2d[n][1], 4)}
                    for n in ch_names if n in pos2d],
        "temporal_response_freqs": r(tfreqs),
        "curated": curated,
    }
    _dump(os.path.join(out_dir, "meta.json"), meta)

    # ---- model.json
    _dump(os.path.join(out_dir, "model.json"), {
        "spatial_W": r(spatial_W), "temporal_K": r(temporal_K),
        "temporal_response": r(tresp),
        "dense_W": r(dense_W), "dense_b": r(fcB),
        "n_pooled": int(model.n_pooled),
    })

    # ---- per-trial files (only the curated set — size discipline)
    spatial = st["spatial"].cpu().numpy()[:, :, 0, :]     # (N,8,T)
    temporal = st["temporal"].cpu().numpy()[:, :, 0, :]   # (N,8,T)
    squared = st["squared"].cpu().numpy()[:, :, 0, :]     # (N,8,T)
    pooled = st["flat"].cpu().numpy()                     # (N,n_pooled) — component-major, reshape [8][38] for UI
    for i in sorted(set(chosen)):
        contrib = pooled[i][:, None] * dense_W            # [n_pooled][2]
        _dump(os.path.join(out_dir, "trials", f"trial_{i:02d}.json"), {
            "id": f"trial_{i:02d}", "label": int(y[i]), "pred": int(preds[i]),
            "correct": bool(preds[i] == y[i]), "confidence": round(float(probs[i].max()), 4),
            "split": split_of.get(int(i), "all"),
            "raw_display": r(X[i, disp_idx, :]),          # 8 channels only
            "spatial_out": r(spatial[i]), "temporal_out": r(temporal[i]),
            "squared": r(squared[i]), "pooled": r(pooled[i]),
            "contributions": r(contrib), "logits": r(logits[i]), "probs": r(probs[i]),
        })

    # ---- consistency check: logits == pooled·dense_W + b  (exported arrays are self-consistent)
    recon = pooled @ dense_W + fcB
    max_err = float(np.max(np.abs(recon - logits)))
    return dict(curated=curated, chosen=sorted(set(chosen)), max_logit_err=max_err,
                n_pooled=int(model.n_pooled), acc=float((preds == y).mean()))

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, default=1, help="PhysioNet subject id (within-subject)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", type=str, default="./assets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, no download — verifies plumbing")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="compute device; 'auto' picks cuda->mps->cpu. Use 'cpu' if MPS crashes "
                         "(torch<2.1 can SIGBUS on Apple Silicon during this model's backward pass).")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")

    if args.smoke:
        X, y, ch_names, pos2d, sfreq = load_smoke_data(seed=args.seed)
        epochs = min(args.epochs, 60)
    else:
        print(f"loading PhysioNet subject {args.subject} (runs 4/8/12, imagined L/R fist)...")
        X, y, ch_names, pos2d, sfreq = load_real_data(args.subject)
        epochs = args.epochs
    print(f"data: X={X.shape}  y balance={np.bincount(y)}  sfreq={sfreq}")

    # within-subject 60/20/20, stratified by label (small sets need this)
    tr, va, te = stratified_split(y, seed=args.seed)
    split_of = {**{int(i): "train" for i in tr}, **{int(i): "val" for i in va},
                **{int(i): "test" for i in te}}
    print(f"split: train={len(tr)} (bal {np.bincount(y[tr])})  "
          f"val={len(va)} (bal {np.bincount(y[va]) if len(va) else []})  "
          f"test={len(te)} (bal {np.bincount(y[te]) if len(te) else []})")

    # [LEAK FIX] fit the per-channel z-score on TRAINING trials only, then apply to all
    X = zscore_fit_apply(X, tr)

    Model = build_model_module()
    model = Model(n_chan=X.shape[1], t_len=X.shape[2]).to(device)
    print(f"model: n_pooled={model.n_pooled}")

    model, best_va = train(model, X[tr], y[tr], X[va], y[va], device, epochs=epochs)

    model.eval()
    with torch.no_grad():
        te_acc = (model(torch.tensor(X[te][:, None], device=device)).argmax(1).cpu().numpy() == y[te]).mean() \
            if len(te) else float("nan")
    print(f"best val_acc={best_va:.3f}  test_acc={te_acc:.3f}")

    info = export(model, X, y, ch_names, pos2d, sfreq, args.out, device, split_of=split_of)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"\nexported -> {args.out}")
    print(f"  logit reconstruction max err: {info['max_logit_err']:.2e}  (should be ~0)")
    print(f"  n_pooled={info['n_pooled']}  full-set acc={info['acc']:.3f}")
    print(f"  total asset size: {total/1024:.0f} KB  (target < ~2 MB)")
    print(f"  curated candidates (edit meta.json to override):")
    for k, v in info["curated"].items():
        print(f"    {k:18s} {v}")
    print("\nNEXT: eyeball the confidently_wrong + artifact trials; they carry the pedagogy (§8).")

if __name__ == "__main__":
    main()
