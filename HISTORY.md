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

---

## 2026-06-20 — Scaling Up Flow Matching

### Changes
- `hidden_units`: `[1024, 1024, 1024]` → `[1024, 1024, 1024, 1024]` (4 layers, ~3M params)
- `num_data_points`: 50k → 100k
- `train_iters`: 5e5 → 5e6

### Bug fixed
`generate_points.py`: off-by-one error when sampling — `int(pt[0] * w)` can equal `w` when `pt[0]=1.0`, causing `IndexError: image index out of range`. Only triggered reliably at 100k+ samples. Fixed by clamping: `min(int(pt[0] * w), w - 1)`.

---

