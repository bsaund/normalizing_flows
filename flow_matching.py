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
    'train_iters': 5e5,
    'num_data_points': 50000,
    'hidden_units': [1024, 1024, 1024],
    'time_embed_dim': 64,    # sinusoidal time embedding dimension
    'visualize_data': False,
    'print_period': 1000,
    'plot_period': 2000,
    'plot_axis_limit': 3.0,
    'ode_steps': 100,        # Euler steps used when integrating the ODE
    'plot_t_steps': 8,       # number of time slices shown in the trajectory plot
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
    MLP that predicts the velocity v_θ(x, t) at position x and time t.

    Input:  [x, sin/cos-embed(t)]  — 2D position + sinusoidal time embedding
    Output: [v₀, v₁]              — 2D velocity

    Sinusoidal embedding gives the network explicit multi-frequency access to t,
    making it much easier to learn time-varying velocity fields compared to
    appending raw t as a single scalar.
    """
    def __init__(self, hidden_units=None, time_embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        hidden_units   = hidden_units   or settings['hidden_units']
        self.embed_dim = time_embed_dim or settings['time_embed_dim']
        self.net = tf_keras.Sequential(
            [tf_keras.layers.Dense(h, activation='relu') for h in hidden_units]
            + [tf_keras.layers.Dense(2)]
        )

    def call(self, x, t):
        """
        x: (batch, 2)
        t: (batch, 1)  or scalar broadcast-able to (batch, 1)
        """
        t   = tf.broadcast_to(tf.reshape(t, (-1, 1)), (tf.shape(x)[0], 1))
        t_e = sinusoidal_time_embedding(t, self.embed_dim)  # (batch, embed_dim)
        xt  = tf.concat([x, t_e], axis=-1)                  # (batch, 2+embed_dim)
        return self.net(xt)

    @tf.function
    def train_step(self, x1_batch, optimizer):
        """
        One CFM training step.

        Given a batch of data points x₁:
          1. Sample noise x₀ ~ N(0, I)
          2. Sample time   t  ~ Uniform(0, 1)
          3. Interpolate   x_t = (1-t)·x₀ + t·x₁
          4. Target vel    u_t = x₁ - x₀   (constant along the straight path)
          5. Loss          MSE(v_θ(x_t, t), u_t)
        """
        batch_size = tf.shape(x1_batch)[0]
        x0  = tf.random.normal(tf.shape(x1_batch))
        t   = tf.random.uniform((batch_size, 1))
        x_t = (1.0 - t) * x0 + t * x1_batch
        u_t = x1_batch - x0

        with tf.GradientTape() as tape:
            v_pred = self(x_t, t, training=True)
            loss   = tf.reduce_mean(tf.square(v_pred - u_t))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def trajectory(self, x0, ode_steps=None):
        """
        Integrate a fixed set of starting points x0 from t=0 to t=1 and
        return snapshots at evenly-spaced time slices (for visualization).

        Returns a list of (n, 2) numpy arrays, one per time slice.
        """
        ode_steps      = ode_steps or settings['ode_steps']
        n_slices       = settings['plot_t_steps']
        dt             = 1.0 / ode_steps
        snapshot_every = max(1, ode_steps // (n_slices - 1))

        x         = tf.constant(x0, dtype=tf.float32)
        snapshots = [x.numpy()]
        for step in range(ode_steps):
            t = tf.fill((tf.shape(x)[0], 1), step * dt)
            x = x + self(x, t, training=False) * dt
            if (step + 1) % snapshot_every == 0 or step == ode_steps - 1:
                snapshots.append(x.numpy())

        return snapshots[:n_slices]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_velocity_grid(model, t_val, lim, grid_n=20):
    """
    Evaluate the velocity field on a regular (grid_n x grid_n) grid at time t_val.
    Returns (gx, gy, vx, vy) — all shape (grid_n, grid_n).
    """
    lin  = np.linspace(-lim, lim, grid_n)
    gx, gy = np.meshgrid(lin, lin)                        # (grid_n, grid_n)
    pts  = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    t_tf = tf.fill((pts.shape[0], 1), float(t_val))
    v    = model(tf.constant(pts), t_tf, training=False).numpy()
    vx   = v[:, 0].reshape(grid_n, grid_n)
    vy   = v[:, 1].reshape(grid_n, grid_n)
    return gx, gy, vx, vy


def plot_trajectory(model, save_path=None, step=None):
    """
    Show the ODE trajectory at evenly-spaced time slices t=0…1, with the
    learned velocity field overlaid as a quiver plot on each panel.
    All panels share fixed axis bounds for stable video frames.
    """
    lim      = settings['plot_axis_limit']
    n_slices = settings['plot_t_steps']

    # Fixed seed — same starting points every frame for a jitter-free video
    tf.random.set_seed(42)
    x0 = tf.random.normal((4000, 2)).numpy()

    snapshots = model.trajectory(x0, ode_steps=settings['ode_steps'])

    rows = 2
    cols = (n_slices + rows - 1) // rows
    f, arr = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    step_str = 'Step {:,}'.format(step) if step is not None else ''
    info = 'Flow Matching (CFM)  |  {}  |  hidden={}  lr={}→{}  batch={}'.format(
        step_str, settings['hidden_units'],
        settings['learning_rate_start'], settings['learning_rate_end'],
        settings['batch_size'])
    f.suptitle(info, fontsize=11, fontweight='bold')

    X0 = snapshots[0]
    for idx, (ax, snap) in enumerate(zip(arr.flat, snapshots)):
        t_val = idx / (n_slices - 1)

        # --- scatter: particles colored by starting quadrant ---
        q = np.stack([X0[:, 0] < 0, X0[:, 1] < 0], axis=1)
        colors = {(True,  True):  'red',
                  (False, True):  'green',
                  (True,  False): 'blue',
                  (False, False): 'black'}
        for (qx, qy), color in colors.items():
            mask = (q[:, 0] == qx) & (q[:, 1] == qy)
            ax.scatter(snap[mask, 0], snap[mask, 1], s=5, color=color, alpha=0.4)

        # --- quiver: velocity field at this time slice ---
        gx, gy, vx, vy = make_velocity_grid(model, t_val, lim, grid_n=20)
        dt = 1.0 / (n_slices - 1)
        ax.quiver(gx, gy, vx * dt, vy * dt,
                  angles='xy', scale_units='xy', scale=1,
                  color='gray', alpha=0.7, width=0.003)

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_aspect('equal')
        ax.set_title('t = {:.2f}'.format(t_val))

    for ax in list(arr.flat)[len(snapshots):]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        f.savefig(save_path, dpi=100)
        plt.close(f)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def make_checkpoint_manager(model, optimizer, checkpoint_dir='checkpoints_fm'):
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

    os.makedirs('training_progress_fm', exist_ok=True)
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
        loss = model.train_step(next(itr), optimizer)
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
                            save_path='training_progress_fm/step_{:07d}.png'.format(i),
                            step=i)

    return loss.numpy() if loss is not None else float('nan')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def create_dataset():
    pts = create_points('BRAD.png', settings['num_data_points'])
    if settings['visualize_data']:
        visualize_data(pts)
    ds = tf.data.Dataset.from_tensor_slices(pts)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=len(pts))
    ds = ds.prefetch(3 * settings['batch_size'])
    ds = ds.batch(settings['batch_size'])
    return ds, pts


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
    model(tf.zeros((1, 2)), tf.zeros((1, 1)))
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
        for path in ('checkpoints_fm', 'training_progress_fm'):
            if os.path.exists(path):
                shutil.rmtree(path)
                print("Deleted: {}".format(path))

    train_and_run_model()
