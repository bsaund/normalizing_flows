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

import math
import os
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
    'learning_rate': 3e-4,
    'train_iters': 2e5,
    'hidden_units': [512, 512],
    'time_embed_dim': 64,    # sinusoidal time embedding dimension
    'visualize_data': False,
    'print_period': 1000,
    'plot_period': 500,
    'plot_axis_limit': 5.0,
    'ode_steps': 5,        # Euler steps used when sampling
    'plot_t_steps': 8,       # number of time slices shown in the layer plot
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t, dim):
    """
    Map scalar time t ∈ [0,1] to a (batch, dim) sinusoidal embedding.

    Uses log-spaced frequencies so the network can distinguish both coarse
    (slow) and fine (fast) time variation.  Same idea as positional encodings
    in transformers.

    t:   (batch, 1) float32
    dim: even integer — output size (dim/2 sin + dim/2 cos components)
    """
    assert dim % 2 == 0
    half = dim // 2
    # Frequencies: 1, ..., 1000  in log scale
    freqs = tf.exp(
        -math.log(1000.0) * tf.cast(tf.range(half), tf.float32) / (half - 1)
    )                                          # (half,)
    angles = t * freqs                         # (batch, half)
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
        hidden_units    = hidden_units    or settings['hidden_units']
        self.embed_dim  = time_embed_dim  or settings['time_embed_dim']
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
        x0 = tf.random.normal(tf.shape(x1_batch))
        t  = tf.random.uniform((batch_size, 1))

        x_t   = (1.0 - t) * x0 + t * x1_batch
        u_t   = x1_batch - x0          # target velocity (independent of t)

        with tf.GradientTape() as tape:
            v_pred = self(x_t, t, training=True)
            loss   = tf.reduce_mean(tf.square(v_pred - u_t))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def sample(self, n=2000, ode_steps=None):
        """
        Generate n samples by integrating dx/dt = v_θ(x, t) from t=0 to t=1
        using the Euler method.

        Returns a numpy array of shape (n, 2).
        """
        ode_steps = ode_steps or settings['ode_steps']
        dt = 1.0 / ode_steps

        x = tf.random.normal((n, 2))
        for step in range(ode_steps):
            t = tf.fill((n, 1), step * dt)
            x = x + self(x, t, training=False) * dt
        return x.numpy()

    def trajectory(self, x0, ode_steps=None):
        """
        Integrate a fixed set of starting points x0 and return snapshots at
        evenly-spaced time slices (for visualization).

        Returns a list of (n, 2) numpy arrays, one per time slice.
        """
        ode_steps = ode_steps or settings['ode_steps']
        n_slices  = settings['plot_t_steps']
        dt        = 1.0 / ode_steps
        # Which steps to snapshot (always include t=0 and t=1)
        snapshot_every = max(1, ode_steps // (n_slices - 1))

        x = tf.constant(x0, dtype=tf.float32)
        snapshots = [x.numpy()]
        for step in range(ode_steps):
            t = tf.fill((tf.shape(x)[0], 1), step * dt)
            x = x + self(x, t, training=False) * dt
            if (step + 1) % snapshot_every == 0 or step == ode_steps - 1:
                snapshots.append(x.numpy())

        # Trim to exactly n_slices
        return snapshots[:n_slices]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectory(model, save_path=None, step=None):
    """
    Show the ODE trajectory at evenly-spaced time slices.

    Analogous to plot_layers() in normalizing_flows.py, but instead of
    bijector layers the panels show t = 0, ..., 1 along the flow.

    All panels share fixed axis bounds for stable video frames.
    """
    lim = settings['plot_axis_limit']
    n_slices = settings['plot_t_steps']

    # Fixed seed — same starting points every frame for a jitter-free video
    tf.random.set_seed(42)
    x0 = tf.random.normal((4000, 2)).numpy()

    snapshots = model.trajectory(x0, ode_steps=settings['ode_steps'])

    rows = 2
    cols = (n_slices + rows - 1) // rows
    f, arr = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    step_str = 'Step {:,}'.format(step) if step is not None else ''
    info = 'Flow Matching (CFM)  |  {}  |  hidden={}  embed={}  lr={}  batch={}'.format(
        step_str, settings['hidden_units'], settings['time_embed_dim'],
        settings['learning_rate'], settings['batch_size'])
    f.suptitle(info, fontsize=11, fontweight='bold')

    X0 = snapshots[0]
    for idx, (ax, snap) in enumerate(zip(arr.flat, snapshots)):
        X = snap
        t_val = idx / (n_slices - 1)
        # Color by quadrant of the *starting* noise point
        q = np.stack([X0[:, 0] < 0, X0[:, 1] < 0], axis=1)
        colors = {(True,  True):  'red',
                  (False, True):  'green',
                  (True,  False): 'blue',
                  (False, False): 'black'}
        for (qx, qy), color in colors.items():
            mask = (q[:, 0] == qx) & (q[:, 1] == qy)
            ax.scatter(X[mask, 0], X[mask, 1], s=5, color=color)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_aspect('equal')
        ax.set_title('t = {:.2f}'.format(t_val))

    # Hide any unused panels
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
    ckpt    = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
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
        x1_batch = next(itr)
        loss     = model.train_step(x1_batch, optimizer)
        global_step.assign(i)

        if i % print_period == 0:
            loss_val = loss.numpy()
            print("{} loss: {:.6f}, {:.1f}s".format(i, loss_val, time() - start))
            if np.isnan(loss_val):
                print("NaN loss — stopping.")
                break

        if i % plot_period == 0:
            manager.save()
            save_path = 'training_progress_fm/step_{:07d}.png'.format(i)
            plot_trajectory(model, save_path=save_path, step=i)

    return loss.numpy() if loss is not None else float('nan')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def create_dataset():
    pts = create_points('BRAD.png', 10000)

    if settings['visualize_data']:
        visualize_data(pts)

    ds = tf.data.Dataset.from_tensor_slices(pts)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=9000)
    ds = ds.prefetch(3 * settings['batch_size'])
    ds = ds.batch(settings['batch_size'])
    return ds, pts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_settings():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU: {}".format(gpus[0].name))
    else:
        print("WARNING: No GPU detected, training on CPU")
    print("Using settings:")
    for k, v in settings.items():
        print('  {}: {}'.format(k, v))


def train_and_run_model(display=True):
    print_settings()

    ds, pts = create_dataset()

    model = VelocityField()
    # Build the network by running one forward pass
    model(tf.zeros((1, 2)), tf.zeros((1, 1)))
    model.summary()

    optimizer = tf_keras.optimizers.Adam(
        learning_rate=settings['learning_rate'], jit_compile=False)

    loss = train(model, ds, optimizer)
    print("Final loss: {:.6f}".format(loss))

    if display:
        plot_trajectory(model)

    return loss


if __name__ == '__main__':
    train_and_run_model()
