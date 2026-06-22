# Experiment History

Lab notebook recording experiments, findings, and parameter choices.

---

## 2026-06-19 — Project Revival & Modernization

### Goal
Get the original 2020 RealNVP code running on a new machine (Python 3.12, Ubuntu 24.04, GTX 1070).

### Findings
- **TF 2.16+ ships Keras 3**, which breaks `tensorflow_probability`. Fix: install `tensorflow-probability[tf]` which pulls in `tf-keras` (Keras 2). Use `tf_keras` throughout instead of `tf.keras`.
- **`real_nvp_default_template`** uses legacy TF1 layers, incompatible with modern Keras. Replaced with a `tf_keras.Sequential` factory (`nvp_shift_and_log_scale_fn`).
- **`masked_autoregressive_default_template`** similarly replaced with `tfb.AutoregressiveNetwork`.
- **NVIDIA driver 470 → 580**: driver 470 only supports CUDA 11.4; TF 2.21 needs CUDA 12. Upgraded to driver 580 (latest for Pascal/GTX 10-series; 590+ drops Pascal support).
- **`tensorflow[and-cuda]` + `tensorflow-cpu` conflict**: both packages provide the same `tensorflow/python/` namespace. `tensorflow-cpu` is CPU-only and "wins" if both are installed. Solution: install `tensorflow-cpu` (provides Python code) plus the `nvidia-*` CUDA packages from `tensorflow[and-cuda]`, and set `LD_LIBRARY_PATH` in the venv activate script to point at the pip-installed CUDA libs.
- **WRAPT + Python 3.12**: set `WRAPT_DISABLE_EXTENSIONS=1` env var to suppress C-extension warnings.

### Infrastructure added
- `requirements.txt` pinned to `tensorflow[and-cuda]==2.21`, `tensorflow-probability[tf]==0.25`
- `.gitignore` for `.venv/`, `checkpoints/`, `training_progress/`
- Checkpoint save/restore with persistent `global_step` (filenames continuous across restarts)
- `--start-new` CLI flag to wipe checkpoints and progress images
- Periodic PNG saves every `plot_period` steps to `training_progress/`
- `make_video.py` — converts training PNGs to MP4 timelapse via ffmpeg
- Seeded RNG in visualization so frames are jitter-free for video

---

## 2026-06-19 — RealNVP Training Fixes

### Problem: NaN loss after ~1000 steps
- `BatchNormalization` bijectors caused gradient explosion / NaN early in training.
- **Fix: removed all `BatchNormalization` bijectors.** Flow still works without them.

### Problem: Training extremely slow (~7 min per 1000 steps), low GPU utilization (~8%)
Two bottlenecks identified:
1. **`.numpy()` sync every step**: calling `loss.numpy()` forces a CPU-GPU synchronization before the next batch starts. Fixed by only syncing at print intervals.
2. **No `@tf.function` on `train_step`**: every op was dispatched individually. Adding `@tf.function` compiles the full forward+backward pass into one graph call.
- Result: GPU utilization jumped to ~98%, ~1000 steps in seconds.

### Problem: Optimizer retracing warnings
- `tf_keras.optimizers.Adam` uses XLA compilation internally and retraces for each unique variable shape. With 8 bijector networks × multiple layers, this triggered constantly.
- **Fix: `jit_compile=False`** in the Adam constructor.

### Working settings (RealNVP)
```python
settings = {
    'batch_size': 1500,
    'method': 'NVP',
    'num_bijectors': 8,
    'learning_rate': 1e-4,
    'train_iters': 2e5,
    'plot_period': 500,
}
```

---

## 2026-06-19 — Flow Matching Implementation

### Motivation
Implement Conditional Flow Matching (CFM)

### Architecture: `flow_matching.py`
- Single MLP `VelocityField(x, t) → velocity`
- Input: 2D position + sinusoidal time embedding (64-dim)
- Training: sample random `(x₀, x₁, t)`, compute `x_t = (1-t)x₀ + tx₁`, minimize `||v_θ(x_t,t) - (x₁-x₀)||²`
- Sampling: Euler ODE integration from t=0→1 (`ode_steps=100`)
- Visualization: 8 trajectory panels (t=0…1) with velocity field quiver overlay

### Sinusoidal time embedding — iteration
- Had to look this up. Cursor got it wrong at first. But core idea is bascially a fourier transform of t. 

### Experiment: Optimal Transport CFM
**Hypothesis**: random pairing causes velocity cancellation at central points (trajectories toward B, R, A, D all cross through the same intermediate regions, their average velocity points toward the centroid = "box" shape instead of letter strokes).

**Implementation**: `scipy.optimize.linear_sum_assignment` to find min-cost assignment of noise→data within each batch chunk.

**Findings**:
- OT-CFM loss dropped from 0.35 → 0.029 in 1000 steps (vs random pairing stuck at ~0.77 for 50k+ steps)
- **However**: OT matching runs on CPU — very slow (~41s per 1000 steps vs seconds for random pairing)
- **Key insight**: OT is NOT necessary. With sufficient training, the velocity field at t≈1 can be arbitrarily sharp because `x_t` is already near the data manifold. The "box" result was due to insufficient training and capacity, not a fundamental limitation of random pairing.
- **Decision**: removed OT code to keep implementation clean and fast.

### Debugging: why stray points outside BRAD?
- Diagnostic: evaluated velocity at `(0,0)` at t=0 → velocity ≈ 0.02 (near zero)
- This IS the correct mean velocity — at the center of the bounding box, gradients toward B, R, A, D roughly cancel. The model is working correctly.
- Resolution: more capacity + more training + LR schedule brings `t≈1` velocities sharp enough to concentrate on letter strokes.

### Successfully reproduced BRAD with these settings
```python
settings = {
    'batch_size': 1500,
    'learning_rate_start': 3e-4,
    'learning_rate_end': 1e-6,     # cosine decay
    'train_iters': 5e5,
    'num_data_points': 50000,
    'hidden_units': [1024, 1024, 1024],
    'time_embed_dim': 64,
    'ode_steps': 100,
}
```

---

## 2026-06-20 — Conditioning Experiment

### Motivation
The core value of a generative model is not just "generate BRAD" but "generate BRAD *given some context*". The simplest possible conditioning signal is a 2D spatial offset: the model should learn that if `obs = (dx, dy)`, the output should be BRAD shifted by that offset.

This is a minimal but complete end-to-end test of the conditioning pipeline — if it works, we know the observation is being propagated correctly through training and inference.

### Changes

**Architecture (`flow_matching.py`)**
- `VelocityField.call(x, t, obs)` — observation concatenated after the time embedding: input is `[x(2), t_embed(64), obs(2)]` = 68D total. Only 2 extra input dimensions; everything else unchanged.
- New `obs_dim: 2` and `obs_range: 1.0` settings.

**Dataset**
- For each BRAD point `p`, sample a random offset `o ~ U(-1, 1)²`.
- Store `(x1 = p + o, obs = o)` pairs.
- At training time the model sees shifted BRAD points alongside the offset that caused the shift. It must learn the velocity field conditioned on that offset.

**Visualization**
- `plot_trajectory` now shows 4 rows, one per conditioning value: `(0,0)`, `(1,0)`, `(0,1)`, `(-1,-1)`.
- Row 0 should look like the original BRAD; other rows should show it shifted accordingly.
- `plot_axis_limit` bumped to `4.0` to accommodate the shift.

### Interactive Applet (`interactive_viz.py`)
New standalone script — loads the trained checkpoint and opens a live matplotlib window:
- 5 scatter panels across the top showing the ODE trajectory at t=0, 0.25, 0.5, 0.75, 1.0
- Two sliders (`obs x`, `obs y`) below
- Every slider drag re-runs the ODE integration (30 Euler steps × 2000 points on GPU) and updates all panels in real time

```bash
python interactive_viz.py                   # default
python interactive_viz.py --ode-steps 10    # faster, rougher
python interactive_viz.py --n-points 5000   # more points
```

Expected behavior: drag `obs x` from 0 → 1 and BRAD glides one unit right in real time.

---

## 2026-06-20 — Conditioning on Dataset Class (BRAD / KATIE) + Spatial Offset

### Motivation
The next step toward a general conditioned generative model: the model should learn to generate *different distributions* based on a discrete observation, not just a spatial shift. Adding a second PNG (`KATIE.png`) and conditioning on a 1-hot class selector tests whether the model can route flow trajectories to entirely different target shapes.

### Observation vector layout
`obs = [dx, dy, is_brad, is_katie]`  — total `obs_dim = 4`

| Dims | Meaning |
|------|---------|
| 0–1 | Spatial offset `(dx, dy) ~ U(-1, 1)²` applied to source points |
| 2   | 1-hot: `1` if target is BRAD, `0` otherwise |
| 3   | 1-hot: `1` if target is KATIE, `0` otherwise |

One-hot encoding chosen over a single integer class index because it gives the model explicit independent input dimensions per class, avoids ordinal bias, and is trivially extensible to N classes.

### Dataset
- 50% of samples from `BRAD.png`, 50% from `KATIE.png` (n/2 each)
- Each sample: `x1 = source_point + (dx, dy)`, `obs = [dx, dy, class_one_hot]`
- Total `num_data_points = 100k` → 50k per class

### Result
The model successfully learns to:
- Generate **BRAD** letter strokes when `obs[2]=1, obs[3]=0`
- Generate **KATIE** letter strokes when `obs[2]=0, obs[3]=1`
- Apply the correct spatial shift `(dx, dy)` in both cases

The `PLOT_CONDITIONS` in `flow_trajectory` shows 4 rows: BRAD/KATIE × centered/shifted-right, making both classes visible in every training progress image.

### Interactive Visualization Update (`interactive_viz.py`)
- Added a `RadioItems` selector (BRAD / KATIE) at the top of the page
- Clicking the radio button immediately re-runs inference with the appropriate 1-hot class bit and updates all 5 trajectory panels
- The title now shows the active class name in bold

### Text-to-Points Infrastructure (`generate_points.py`, `text_preview.py`)
Added support for generating points from any rendered text string, not just hardcoded PNGs:
- `render_text_image(text, font_size)` — renders text to a PIL image using the best available bold TTF font (found via `matplotlib.font_manager`)
- `create_points_from_text(text, num_points)` — samples from dark pixels of the rendered image; same coordinate normalization as `create_points()`
- `text_preview.py` — Dash app at port 8051 for interactively previewing any word and its sampled point cloud before committing to training

### Text-to-Points Infrastructure (`generate_points.py`, `text_preview.py`)
New capability: generate training point clouds from arbitrary text strings without needing pre-made PNG files.

| Function | Description |
|---|---|
| `render_text_image(text, font_size, padding)` | Renders text to a PIL `Image` using the best available bold TTF font (via `matplotlib.font_manager`). Falls back to the PIL built-in if no system font is found. |
| `create_points_from_text(text, num_points, font_size)` | Samples `num_points` from dark pixels of the rendered image; uses the same coordinate normalisation as `create_points()`. |
| `text_preview.py` | Dash app at port 8051 for interactively previewing any word and its sampled point cloud before committing to training. Controls: text input, font-size slider, num-points slider. |

This infrastructure removes the manual step of creating PNGs and opens the door to conditioning on arbitrary words.

---

## 2026-06-22 — Transformer Flow Matching: Single-Letter and Two-Letter Experiments

### Single-letter PoC (A–Z)
Architecture: d_model=128, n_layers=2, n_heads=2 (~426k params).
Vocabulary: all 26 uppercase letters.
Result: **converged cleanly in ~1300 steps**. Letters were clearly recognisable very early.
This confirms the core architecture works — the coord token attends to the char token,
and the model correctly routes each Gaussian point toward the appropriate letter stroke.

### Two-letter experiment
Architecture: d_model=128, n_layers=3, n_heads=2 (~625k params).
Vocabulary: 200 random 2-char pairs, 500 pts/word → 100k static samples.
Settings: train_iters=2e6, cosine LR 3e-4 → 1e-6.

**Observed training trajectory:**

| Steps | Quality |
|---|---|
| 20k | Letters forming, fairly crisp |
| 50k | Peak crispness |
| 100k | Crisper but missing some strokes (sampling bias) |
| 150k | Starting to blur |
| 400k | Letters no longer recognisable |

The model found a good solution early then progressively degraded.

### Root-cause analysis

**1. Learning rate too high for too long (primary cause)**

The cosine schedule decays over `train_iters=2e6` steps.
At the observed peak (~50k steps) the LR is still at ~99% of its starting value (3e-4).
The model found a good solution, then continued receiving large gradient updates for
another 350k steps — eroding the learned representation. Classic "trained past the
optimum" failure.

```
step 50k  →  LR ≈ 2.98e-4  (99% of max)   ← peak quality here
step 400k →  LR ≈ 2.58e-4  (86% of max)   ← quality gone
```

**2. Dataset over-repetition (secondary cause)**

100k static training samples × 2M steps × 1500 batch = each sample seen ~30,000 times.
With a fixed, small dataset, the optimizer has nothing left to learn and begins oscillating
around the noise in the data.

**3. Architecture probably sufficient** — the single-letter result shows the transformer
can represent individual character shapes. Two letters should be a composition of that
knowledge, not a qualitatively harder task for the architecture.

### Fixes to try next
- **Train/validation split**: hold out a set of words unseen during training (e.g. 20% of
  the 200-word vocab).  Track validation loss alongside training loss.  If val loss stops
  improving while train loss keeps falling, that's the clean stopping signal — save that
  checkpoint.  This is more principled than visual inspection and would have caught the
  degradation automatically.
- Reduce `train_iters` to ~1e5 (match observed peak) or use early stopping on val loss.
- Regenerate offsets dynamically each epoch instead of baking them in at dataset creation
  (increases effective diversity from 100k to essentially infinite, delays over-repetition).
- Increase dataset size (more words, more pts/word) as a simpler alternative.

### Status: paused
Experiment paused at this point to let the project rest.  Next logical step is to add a
train/validation split and use it to find the optimal stopping point reliably.

---

## 2026-06-20 — Scaling Up Flow Matching

### Changes
- `hidden_units`: `[1024, 1024, 1024]` → `[1024, 1024, 1024, 1024]` (4 layers, ~3M params)
- `num_data_points`: 50k → 100k
- `train_iters`: 5e5 → 5e6

### Bug fixed
`generate_points.py`: off-by-one error when sampling — `int(pt[0] * w)` can equal `w` when `pt[0]=1.0`, causing `IndexError: image index out of range`. Only triggered reliably at 100k+ samples. Fixed by clamping: `min(int(pt[0] * w), w - 1)`.

---

## 2026-06-21 — GPU Sinkhorn Optimal Transport CFM

### Motivation
Basic CFM pairs each noise sample `x0[i]` with a **random** data point `x1[j]`.
Different x0 samples targeting the same region of BRAD/KATIE can be assigned to
far-away data points, forcing the velocity field to learn crossing, tangled paths.
Optimal Transport CFM (OT-CFM) fixes this by pairing `x0` and `x1` to minimise
total squared Euclidean distance, so trajectories cross as little as possible.

### Previous attempt (CPU linear_sum_assignment)
An earlier prototype used `scipy.optimize.linear_sum_assignment` (Hungarian
algorithm) for exact OT.  It was removed because:
1. Hungarian runs on CPU → bottleneck in every training step
2. O(n³) complexity makes it impractical for batch size 1500

### This experiment: GPU Sinkhorn
Sinkhorn is an entropic-regularised approximation to OT that runs entirely on
GPU as a sequence of matrix operations:

```
K = exp(−C / ε)        where  C[i,j] = ||x0[i] − x1[j]||²
repeat n_iters:
  log_u ← log(1/n) − logsumexp(log_K + log_v,  axis=1)
  log_v ← log(1/n) − logsumexp(log_K + log_u,  axis=0)
coupling:  log_P[i,j] = log_u[i] + log_K[i,j] + log_v[j]
pair:      for each x0[i], sample x1[σ(i)] ~ softmax(log_P[i,:])
```

The loop is unrolled at `@tf.function` trace time (2 × `n_iters`
`reduce_logsumexp` ops on a `(B, B)` matrix).  On a GTX 1070 this adds < 1 %
wall-clock overhead over the network forward/backward pass.

### Parameters added to `settings`
| Key | Default | Meaning |
|---|---|---|
| `use_sinkhorn` | `True` | Enable/disable OT pairing |
| `sinkhorn_epsilon` | `0.05` | Regularisation: smaller → sharper OT, more Sinkhorn iters needed |
| `sinkhorn_iters` | `30` | Sinkhorn iterations (unrolled into TF graph at trace time) |

### Design notes
- **Log-domain Sinkhorn** is used throughout to avoid exp-overflow/underflow.
  With ε = 0.05 and 2D data in [−4, 4], the cost matrix reaches ~130, so
  log_K can be as low as −2600 — fine for logsumexp but dangerous for raw exp.
- **Obs reordering**: each data point `x1[j]` carries an `obs[j]` (offset +
  class one-hot).  When x1 is reordered by the OT permutation σ, `obs` is
  reordered by the same permutation so the conditioning stays consistent.
- **Epsilon tradeoff**: small ε (e.g. 0.01) gives near-exact OT but needs
  more iterations and the coupling is sharper (less stochastic). Large ε
  (e.g. 1.0) approaches random pairing. ε = 0.05 is a practical middle ground.

### Results

**Pro — far fewer ODE steps needed for a readable result**

| ODE steps | Random CFM | Sinkhorn OT-CFM |
|---|---|---|
| 1 | noise | recognisable blob |
| 3 | barely anything | readable "BRAD" |
| 5 | first hints of letters | clean letters |
| 20–100 | needed to look good | overkill |

OT reduces trajectory crossings so the paths are straighter; a single Euler
step already lands points close to the letter shapes.  This is the main
theoretical promise of OT-CFM delivered in practice.

**Con — centre-of-distribution bias under spatial offset conditioning**

When `(dx, dy)` shifts BRAD away from the origin, the generated point cloud
does *not* fully follow.  Points cluster toward the centre of the training
distribution rather than faithfully tracking the requested offset.

Root cause: with `epsilon = 0.05` the Sinkhorn coupling is very sharp (nearly
a hard one-to-one assignment).  Gaussian noise samples `x0 ~ N(0, I)` are
densest near the origin.  OT pairs each `x0[i]` with the nearest available
`x1[j]` — which for centred noise is always a near-origin data point,
regardless of what offset `obs` requests.  Noise samples that *should* reach
shifted data points (e.g., BRAD at `dx=2`) are re-routed to nearby unshifted
data points in the same batch.  The velocity field never sees sufficient
training signal for the far-shifted regime, so inference in that region
produces points pulled back toward the origin.

**Takeaways**
- Sinkhorn OT is a real win for *step efficiency* in the unconditioned / lightly
  conditioned case.
- For *spatial offset conditioning* the sharp coupling actively fights the
  conditioning signal.  A softer epsilon (0.1 – 0.5) or abandoning OT pairing
  is preferable when large offsets are part of the observation space.
- The loss metric is not comparable between random-CFM and OT-CFM: OT targets
  are intrinsically shorter vectors, so OT loss floors at ~0.025 while
  random-CFM floors at ~0.35 — neither number predicts generation quality on
  its own.

---

