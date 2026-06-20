#!/usr/bin/env python
"""
Flow Matching on 2D data (BRAD.png or two_moons.png).

Trains a velocity field v_θ(x, t) using the Conditional Flow Matching (CFM)
objective, then samples by integrating the ODE dx/dt = v_θ(x,t) from t=0→1
with Euler steps.

Compare with normalizing_flows.py which uses RealNVP bijectors and
log-likelihood training. Flow matching is simpler: no bijectors, no Jacobians,
just an MSE loss on a randomly sampled (x_t, target_velocity) pair.
"""
from __future__ import print_function

import argparse
import math
import os
import shutil
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "1")

import tensorflow as tf
import tf_keras
import matplotlib.pyplot as plt
import numpy as np
from generate_points import create_points, visualize_data
from time import time

# Allow TF to grow GPU memory incrementally — important on a GPU also driving display.
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)


settings = {
    'batch_size': 1500,
    'learning_rate_start': 3e-4,
    'learning_rate_end':   1e-6,   # cosine decay target
    'train_iters': 5e6,
    'num_data_points': 100000,
    'hidden_units': [1024, 1024, 1024, 1024],
    'time_embed_dim': 64,    # sinusoidal time embedding dimension
    'obs_dim': 2,            # dimension of the conditioning observation vector
    'obs_range': 1.0,        # training offsets drawn from U(-obs_range, obs_range)²
    'visualize_data': False,
    'print_period': 1000,
    'plot_period': 2000,
    'plot_axis_limit': 4.0,
    'ode_steps': 100,        # Euler steps used when integrating the ODE
    'plot_t_steps': 5,       # number of time slices shown per trajectory row
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t, dim):
    """
    Map scalar time t ∈ [0,1] to a (batch, dim) sinusoidal embedding.

    Canonical transformer positional encoding (Vaswani et al., 2017):
      angle_i = t / 10000^(i / half)
    Index 0: divides by 1     → fast sinusoid (fine time structure)
    Index half: divides by 10k → slow sinusoid (coarse time structure)

    t:   (batch, 1) float32
    dim: even integer — output size (dim/2 sin + dim/2 cos components)
    """
    assert dim % 2 == 0
    half = dim // 2

    t = tf.cast(t, tf.float32)

    log_freqs = tf.linspace(0.0, math.log(10000.0), half)  # (half,)
    freqs     = tf.exp(log_freqs)                           # (half,) from 1 to 10000
    freqs     = tf.expand_dims(freqs, axis=0)               # (1, half) for clean broadcasting

    angles = t / freqs                                      # (batch, half)
    return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  # (batch, dim)


class VelocityField(tf_keras.Model):
    """
    Conditioned MLP: v_θ(x, t, obs) → 2D velocity.

    Inputs concatenated before the first layer:
      x   (batch, 2)          — current position
      t   (batch, embed_dim)  — sinusoidal time embedding
      obs (batch, obs_dim)    — conditioning observation (e.g. spatial offset)

    Sinusoidal embedding gives the network explicit multi-frequency access to t,
    making it much easier to learn time-varying velocity fields compared to
    appending raw t as a single scalar.
    """
    def __init__(self, hidden_units=None, time_embed_dim=None, obs_dim=None, **kwargs):
        super().__init__(**kwargs)
        hidden_units   = hidden_units   or settings['hidden_units']
        self.embed_dim = time_embed_dim or settings['time_embed_dim']
        self.obs_dim   = obs_dim if obs_dim is not None else settings['obs_dim']
        self.net = tf_keras.Sequential(
            [tf_keras.layers.Dense(h, activation='relu') for h in hidden_units]
            + [tf_keras.layers.Dense(2)]
        )

    def call(self, x, t, obs):
        """
        x:   (batch, 2)
        t:   (batch, 1)  or scalar broadcast-able to (batch, 1)
        obs: (batch, obs_dim)  — conditioning vector
        """
        t   = tf.broadcast_to(tf.reshape(t, (-1, 1)), (tf.shape(x)[0], 1))
        t_e = sinusoidal_time_embedding(t, self.embed_dim)     # (batch, embed_dim)
        obs = tf.cast(obs, tf.float32)
        inp = tf.concat([x, t_e, obs], axis=-1)                # (batch, 2+embed_dim+obs_dim)
        return self.net(inp)

    @tf.function
    def train_step(self, batch, optimizer):
        """
        One conditioned CFM training step.

        batch: (x1_batch, obs_batch)
          x1_batch: (B, 2)       — target positions (already shifted by obs)
          obs_batch: (B, obs_dim) — conditioning vectors used to shift x1

        Steps:
          1. Sample noise x₀ ~ N(0, I)
          2. Sample time   t  ~ Uniform(0, 1)
          3. Interpolate   x_t = (1-t)·x₀ + t·x₁
          4. Target vel    u_t = x₁ - x₀
          5. Loss          MSE(v_θ(x_t, t, obs), u_t)
        """
        x1_batch, obs_batch = batch
        batch_size = tf.shape(x1_batch)[0]
        x0  = tf.random.normal(tf.shape(x1_batch))
        t   = tf.random.uniform((batch_size, 1))
        x_t = (1.0 - t) * x0 + t * x1_batch
        u_t = x1_batch - x0

        with tf.GradientTape() as tape:
            v_pred = self(x_t, t, obs_batch, training=True)
            loss   = tf.reduce_mean(tf.square(v_pred - u_t))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def trajectory(self, x0, obs, ode_steps=None):
        """
        Integrate starting points x0 from t=0 to t=1 conditioned on obs,
        returning snapshots at evenly-spaced time slices for visualization.

        x0:  (n, 2) float32
        obs: (n, obs_dim) float32 — same conditioning repeated across the batch
        Returns a list of (n, 2) numpy arrays, one per time slice.
        """
        ode_steps      = ode_steps or settings['ode_steps']
        n_slices       = settings['plot_t_steps']
        dt             = 1.0 / ode_steps
        snapshot_every = max(1, ode_steps // (n_slices - 1))

        x   = tf.constant(x0, dtype=tf.float32)
        obs = tf.constant(obs, dtype=tf.float32)
        snapshots = [x.numpy()]
        for step in range(ode_steps):
            t = tf.fill((tf.shape(x)[0], 1), step * dt)
            x = x + self(x, t, obs, training=False) * dt
            if (step + 1) % snapshot_every == 0 or step == ode_steps - 1:
                snapshots.append(x.numpy())

        return snapshots[:n_slices]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_velocity_grid(model, t_val, obs_val, lim, grid_n=20):
    """
    Evaluate the velocity field on a regular (grid_n x grid_n) grid at time
    t_val, conditioned on obs_val.

    obs_val: 1-D array of shape (obs_dim,), broadcast to all grid points.
    Returns (gx, gy, vx, vy) — all shape (grid_n, grid_n).
    """
    lin      = np.linspace(-lim, lim, grid_n)
    gx, gy   = np.meshgrid(lin, lin)                      # (grid_n, grid_n)
    pts      = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    n        = pts.shape[0]
    t_tf     = tf.fill((n, 1), float(t_val))
    obs_tf   = tf.tile(tf.constant([obs_val], dtype=tf.float32), [n, 1])  # (n, obs_dim)
    v        = model(tf.constant(pts), t_tf, obs_tf, training=False).numpy()
    vx       = v[:, 0].reshape(grid_n, grid_n)
    vy       = v[:, 1].reshape(grid_n, grid_n)
    return gx, gy, vx, vy


# Conditioning values shown in the trajectory plot — one row per entry.
# Each entry is [dx, dy] — the offset applied to all BRAD points.
PLOT_CONDITIONS = [
    [ 0.0,  0.0],   # original BRAD
    [ 1.0,  0.0],   # shifted right
    [ 0.0,  1.0],   # shifted up
    [-1.0, -1.0],   # shifted down-left
]


def plot_trajectory(model, save_path=None, step=None):
    """
    Show the ODE trajectory for several conditioning values (rows) at
    evenly-spaced time slices t=0…1 (columns).  Velocity field overlaid
    as a quiver plot.  All panels share fixed axis bounds.
    """
    lim       = settings['plot_axis_limit']
    n_slices  = settings['plot_t_steps']
    conds     = PLOT_CONDITIONS
    n_conds   = len(conds)

    # Fixed seed — same starting points every frame for a jitter-free video
    tf.random.set_seed(42)
    x0_base = tf.random.normal((2000, 2)).numpy()

    step_str = 'Step {:,}'.format(step) if step is not None else ''
    info = ('Flow Matching (conditioned CFM)  |  {}  |  '
            'hidden={}  lr={}→{}  batch={}').format(
        step_str, settings['hidden_units'],
        settings['learning_rate_start'], settings['learning_rate_end'],
        settings['batch_size'])

    f, axes = plt.subplots(n_conds, n_slices,
                           figsize=(4 * n_slices, 4 * n_conds))
    f.suptitle(info, fontsize=10, fontweight='bold')

    # Consistent quadrant coloring based on x0 starting position
    q = np.stack([x0_base[:, 0] < 0, x0_base[:, 1] < 0], axis=1)
    quad_colors = {(True,  True):  'red',
                   (False, True):  'green',
                   (True,  False): 'blue',
                   (False, False): 'black'}

    for row, obs_val in enumerate(conds):
        obs_arr = np.tile(np.array(obs_val, dtype=np.float32), (x0_base.shape[0], 1))
        snapshots = model.trajectory(x0_base, obs_arr, ode_steps=settings['ode_steps'])

        row_axes = axes[row] if n_conds > 1 else axes
        for col, (ax, snap) in enumerate(zip(row_axes, snapshots)):
            t_val = col / (n_slices - 1)

            for (qx, qy), color in quad_colors.items():
                mask = (q[:, 0] == qx) & (q[:, 1] == qy)
                ax.scatter(snap[mask, 0], snap[mask, 1], s=4, color=color, alpha=0.4)

            gx, gy, vx, vy = make_velocity_grid(model, t_val, obs_val, lim, grid_n=15)
            dt = 1.0 / (n_slices - 1)
            ax.quiver(gx, gy, vx * dt, vy * dt,
                      angles='xy', scale_units='xy', scale=1,
                      color='gray', alpha=0.6, width=0.003)

            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_aspect('equal')

            col_label = 't={:.2f}'.format(t_val)
            row_label = 'obs={}'.format(obs_val)
            ax.set_title('{}\n{}'.format(row_label, col_label) if col == 0 else col_label,
                         fontsize=9)

    plt.tight_layout()
    if save_path:
        f.savefig(save_path, dpi=80)
        plt.close(f)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def make_checkpoint_manager(model, optimizer, checkpoint_dir='checkpoints_fm_cond'):
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
    ckpt        = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                      global_step=global_step)
    manager     = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    return ckpt, manager, global_step


def restore_if_available(ckpt, manager):
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print("Restored checkpoint: {}".format(manager.latest_checkpoint))
        return True
    print("No checkpoint found, starting from scratch.")
    return False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, ds, optimizer):
    """
    Training loop.  Prints loss every print_period steps, saves a trajectory
    plot + checkpoint every plot_period steps.
    Uses a persistent global_step so filenames and logs are continuous across restarts.
    """
    print_period = settings['print_period']
    plot_period  = settings['plot_period']
    total_iters  = int(settings['train_iters'])

    os.makedirs('training_progress_fm_cond', exist_ok=True)
    ckpt, manager, global_step = make_checkpoint_manager(model, optimizer)
    restore_if_available(ckpt, manager)

    start_step = int(global_step.numpy())
    if start_step >= total_iters:
        print("Already trained for {} steps, nothing to do.".format(start_step))
        return float('nan')

    print("Resuming from step {}.".format(start_step))
    start = time()
    itr   = iter(ds)
    loss  = None

    for i in range(start_step, total_iters + 1):
        loss = model.train_step(next(itr), optimizer)  # batch is (x1, obs) tuple
        global_step.assign(i)

        if i % print_period == 0:
            loss_val = loss.numpy()
            print("{} loss: {:.6f}, {:.1f}s".format(i, loss_val, time() - start))
            if np.isnan(loss_val):
                print("NaN loss — stopping.")
                break

        if i % plot_period == 0:
            manager.save()
            plot_trajectory(model,
                            save_path='training_progress_fm_cond/step_{:07d}.png'.format(i),
                            step=i)

    return loss.numpy() if loss is not None else float('nan')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def create_dataset():
    """
    Build a dataset of (x1, obs) pairs for conditioned flow matching.

    For each BRAD point `p`, a random 2-D offset `o ~ U(-obs_range, obs_range)²`
    is sampled and stored alongside the shifted point `x1 = p + o`.
    At training time the model must learn: given obs=o, push noise toward BRAD+o.
    """
    brad_pts = create_points('BRAD.png', settings['num_data_points'])
    if settings['visualize_data']:
        visualize_data(brad_pts)

    obs_range = settings['obs_range']
    offsets   = np.random.uniform(-obs_range, obs_range,
                                  size=(len(brad_pts), settings['obs_dim'])).astype(np.float32)
    x1 = (brad_pts + offsets).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x1, offsets))
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=len(brad_pts))
    ds = ds.prefetch(3 * settings['batch_size'])
    ds = ds.batch(settings['batch_size'])
    return ds, brad_pts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_settings():
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU: {}".format(gpus[0].name) if gpus else "WARNING: No GPU detected")
    print("Using settings:")
    for k, v in settings.items():
        print('  {}: {}'.format(k, v))


def train_and_run_model(display=True):
    print_settings()
    ds, pts = create_dataset()

    model = VelocityField()
    # Warmup call to build weights before summary/checkpoint
    dummy_obs = tf.zeros((1, settings['obs_dim']))
    model(tf.zeros((1, 2)), tf.zeros((1, 1)), dummy_obs)
    model.summary()

    lr_schedule = tf_keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=settings['learning_rate_start'],
        decay_steps=int(settings['train_iters']),
        alpha=settings['learning_rate_end'] / settings['learning_rate_start'],
    )
    optimizer = tf_keras.optimizers.Adam(lr_schedule, jit_compile=False)

    loss = train(model, ds, optimizer)
    print("Final loss: {:.6f}".format(loss))

    if display:
        plot_trajectory(model)

    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow Matching on 2D data.')
    parser.add_argument('--start-new', action='store_true',
                        help='Delete existing checkpoints and training images, then start fresh.')
    args = parser.parse_args()

    if args.start_new:
        for path in ('checkpoints_fm_cond', 'training_progress_fm_cond'):
            if os.path.exists(path):
                shutil.rmtree(path)
                print("Deleted: {}".format(path))

    train_and_run_model()
