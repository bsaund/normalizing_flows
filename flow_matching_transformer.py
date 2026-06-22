#!/usr/bin/env python
"""
flow_matching_transformer.py — Flow Matching with a Transformer velocity field.

Conditions on a ≤5-character text string plus a 2-D spatial offset (dx, dy).

Architecture
------------
Token sequence:  [coord_token | char₀ | char₁ | char₂ | char₃ | char₄]
  coord_token : Dense(concat(x, sinusoidal_t, dx, dy)) → d_model
  char_i      : Embedding(char_id) + Embedding(position_i)  → d_model

N pre-norm Transformer blocks (bidirectional self-attention over all 6 tokens).
Velocity output is read from the coord_token position (index 0) after all blocks.

Training data
-------------
N_TRAIN_WORDS random 5-char uppercase words, each represented by POINTS_PER_WORD
samples from create_points_from_text().  Random spatial offsets added per sample.

Run:
    python flow_matching_transformer.py             # train / resume
    python flow_matching_transformer.py --start-new # wipe checkpoint and restart
"""
from __future__ import print_function

import argparse
import math
import os
import random
import shutil
import string
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_keras

from time import time
from generate_points import create_points_from_text

for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)


# ---------------------------------------------------------------------------
# Character tokenisation
# ---------------------------------------------------------------------------

CHARS       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
PAD_IDX     = 26        # also used for space / unknown chars
VOCAB_SIZE  = 27        # 26 letters + PAD
MAX_CHARS   = 5


def encode_word(word: str) -> list:
    """Encode a word to MAX_CHARS character indices (uppercase, padded)."""
    word = word.upper()[:MAX_CHARS]
    indices = [CHAR_TO_IDX.get(c, PAD_IDX) for c in word]
    while len(indices) < MAX_CHARS:
        indices.append(PAD_IDX)
    return indices


def decode_word(indices) -> str:
    """Decode character indices back to a string (for display)."""
    return ''.join(CHARS[i] if i < 26 else ' ' for i in indices)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

settings = {
    'batch_size':   1500,
    'learning_rate_start': 3e-4,
    'learning_rate_end':   1e-6,
    'train_iters':   5e6,
    # Transformer
    'd_model':      256,
    'n_heads':        4,
    'n_layers':       4,
    'time_embed_dim': 64,
    # Data
    'obs_range':      1.0,    # spatial offsets drawn from U(−obs_range, obs_range)²
    'n_train_words':  200,    # distinct words in training vocabulary
    'points_per_word': 500,   # point-cloud samples per word  (total ≈ 100k)
    'font_size':      120,    # PIL render size (larger = more detail but slower render)
    # Logging / visualisation
    'print_period': 1000,
    'plot_period':  2000,
    'plot_axis_limit': 5.0,
    'ode_steps':      20,
    'plot_t_steps':    5,
    # Single-letter proof-of-concept mode (--single-letter flag)
    '_single_letter': {
        'd_model':        128,
        'n_heads':          2,
        'n_layers':         2,
        'n_train_words':   26,   # all 26 letters — no random words needed
        'points_per_word': 1000,
        'train_iters':     5e5,
        'plot_period':     500,
    },
    # Two-letter mode (--two-letter flag)
    '_two_letter': {
        'd_model':        128,
        'n_heads':          2,
        'n_layers':         3,
        'n_train_words':   200,  # random 2-char combos
        'points_per_word': 500,
        'train_iters':     2e5,
        'plot_period':    2000,
    },
}

# Words rendered in every training-progress plot (≤ MAX_CHARS chars each)
PLOT_WORDS        = ['ROBOT', 'BRAIN', 'HELLO', 'WORLD', 'LIGHT']
PLOT_WORDS_SINGLE = ['A', 'E', 'M', 'R', 'S']
PLOT_WORDS_TWO    = ['AB', 'HI', 'OK', 'MR', 'XY']


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (canonical Vaswani et al. 2017)
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t, dim):
    """
    t:   (batch, 1) float32 in [0, 1]
    dim: even integer
    Returns (batch, dim).
    """
    assert dim % 2 == 0
    half      = dim // 2
    t         = tf.cast(t, tf.float32)
    log_freqs = tf.linspace(0.0, math.log(10_000.0), half)   # (half,)
    freqs     = tf.exp(log_freqs)                              # (half,)
    freqs     = tf.expand_dims(freqs, 0)                       # (1, half)
    angles    = t / freqs                                       # (batch, half)
    return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class LayerNorm(tf_keras.layers.Layer):
    """
    Layer normalisation using plain TF ops.
    tf_keras.layers.LayerNormalization uses the FusedBatchNormV3 cuDNN kernel
    for 3-D inputs, which fails on some GPU / cuDNN versions (e.g. GTX 1070).
    This implementation avoids that code path entirely.
    """

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight('gamma', shape=(dim,), initializer='ones')
        self.beta  = self.add_weight('beta',  shape=(dim,), initializer='zeros')

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) * tf.math.rsqrt(var + 1e-6) + self.beta


class TransformerBlock(tf_keras.layers.Layer):
    """
    Pre-norm transformer block:
      x → LN → MultiHeadSelfAttention → + x
        → LN → FFN (GELU)             → + x
    """

    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.ln1  = LayerNorm()
        self.attn = tf_keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
        )
        self.ln2 = LayerNorm()
        self.ff  = tf_keras.Sequential([
            tf_keras.layers.Dense(d_model * 4, activation='gelu'),
            tf_keras.layers.Dense(d_model),
        ])

    def call(self, x, training=False):
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, training=training)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Text-conditioned velocity field
# ---------------------------------------------------------------------------

class TextVelocityField(tf_keras.Model):
    """
    Transformer velocity field conditioned on (text_tokens, dx_dy).

    Token sequence (length 6):
      position 0 — coord token: continuous features (x, t, dx, dy) → d_model
      positions 1–5 — char tokens: Embedding(char) + Embedding(pos)

    Bidirectional self-attention over all 6 tokens.
    Velocity = MLP(output at position 0).
    """

    def __init__(self, d_model=None, n_heads=None, n_layers=None,
                 time_embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        d_model        = d_model        or settings['d_model']
        n_heads        = n_heads        or settings['n_heads']
        n_layers       = n_layers       or settings['n_layers']
        self.embed_dim = time_embed_dim or settings['time_embed_dim']

        # Character + position embeddings for the 5 char tokens
        self.char_embed = tf_keras.layers.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed  = tf_keras.layers.Embedding(MAX_CHARS,  d_model)

        # Coord token: project (x[2] + t_embed[embed_dim] + dx_dy[2]) → d_model
        self.coord_proj = tf_keras.layers.Dense(d_model)

        # Transformer
        self.blocks   = [TransformerBlock(d_model, n_heads, name='block_{}'.format(i))
                         for i in range(n_layers)]
        self.final_ln = LayerNorm()

        # Velocity head: d_model → 2
        self.vel_head = tf_keras.Sequential([
            tf_keras.layers.Dense(d_model, activation='gelu'),
            tf_keras.layers.Dense(2),
        ])

    def call(self, x, t, text_tokens, dx_dy, training=False):
        """
        x:           (B, 2)   current 2-D positions
        t:           (B, 1)   time in [0, 1]
        text_tokens: (B, 5)   int32 character indices (0=A … 25=Z, 26=PAD)
        dx_dy:       (B, 2)   spatial offset conditioning
        Returns:     (B, 2)   velocity
        """
        # --- Coordinate token (position 0 in sequence) ---
        t_e   = sinusoidal_time_embedding(t, self.embed_dim)              # (B, embed)
        cinp  = tf.concat([x, t_e, tf.cast(dx_dy, tf.float32)], axis=-1) # (B, 2+emb+2)
        coord = self.coord_proj(cinp)[:, tf.newaxis, :]                   # (B, 1, d_model)

        # --- Character tokens (positions 1–5) ---
        ch    = self.char_embed(text_tokens)                              # (B, 5, d_model)
        pos   = self.pos_embed(tf.range(MAX_CHARS))                       # (5, d_model)
        chars = ch + pos[tf.newaxis, :, :]                                # (B, 5, d_model)

        # --- Full sequence ---
        seq = tf.concat([coord, chars], axis=1)                           # (B, 6, d_model)

        # --- Transformer ---
        for block in self.blocks:
            seq = block(seq, training=training)
        seq = self.final_ln(seq)

        # --- Velocity from coord token output ---
        return self.vel_head(seq[:, 0, :])                                # (B, 2)

    @tf.function
    def train_step(self, batch, optimizer):
        """
        batch: (x1, tokens, dxdy)
          x1:     (B, 2)   target positions (already shifted by dx_dy)
          tokens: (B, 5)   int32 character indices
          dxdy:   (B, 2)   spatial offset
        """
        x1, tokens, dxdy = batch
        B   = tf.shape(x1)[0]
        x0  = tf.random.normal(tf.shape(x1))
        t   = tf.random.uniform((B, 1))
        x_t = (1.0 - t) * x0 + t * x1
        u_t = x1 - x0

        with tf.GradientTape() as tape:
            v_pred = self(x_t, t, tokens, dxdy, training=True)
            loss   = tf.reduce_mean(tf.square(v_pred - u_t))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def build_vocabulary(mode='full'):
    """
    Build a list of training words.

    mode='single' → all 26 single uppercase letters (A–Z)
    mode='two'    → random 2-char uppercase pairs (includes PLOT_WORDS_TWO)
    mode='full'   → random 5-char words (includes PLOT_WORDS)
    """
    if mode == 'single':
        return list(CHARS)

    n = settings['n_train_words']

    if mode == 'two':
        seed_words = list(PLOT_WORDS_TWO)
        k = 2
    else:
        seed_words = list(PLOT_WORDS)
        k = MAX_CHARS

    vocab = list(seed_words)
    while len(vocab) < n:
        word = ''.join(random.choices(string.ascii_uppercase, k=k))
        if word not in vocab:
            vocab.append(word)
    return vocab[:n]


def create_dataset(mode='full'):
    """
    Pre-render point clouds for each training word, then assemble a
    tf.data.Dataset of (x1, tokens, dxdy) tuples.

    x1:     (2,)   target position = raw_point + (dx, dy)
    tokens: (5,)   int32 character indices (padded to MAX_CHARS)
    dxdy:   (2,)   random spatial offset
    """
    vocab       = build_vocabulary(mode=mode)
    n_pts       = settings['points_per_word']
    obs_range   = settings['obs_range']
    font_size   = settings['font_size']

    x1_list, tok_list, dxdy_list = [], [], []

    print("Rendering {} words ({} pts each) …".format(len(vocab), n_pts))
    for i, word in enumerate(vocab):
        pts    = create_points_from_text(word, n_pts, font_size=font_size)
        tokens = encode_word(word)
        offsets = np.random.uniform(-obs_range, obs_range,
                                    (len(pts), 2)).astype(np.float32)
        x1_list.append((pts + offsets).astype(np.float32))
        tok_list.append(np.tile(tokens, (len(pts), 1)).astype(np.int32))
        dxdy_list.append(offsets)
        if (i + 1) % 50 == 0:
            print("  {}/{} done".format(i + 1, len(vocab)))

    x1_all   = np.concatenate(x1_list,   axis=0)
    tok_all  = np.concatenate(tok_list,  axis=0)
    dxdy_all = np.concatenate(dxdy_list, axis=0)

    print("Dataset: {} samples across {} words.".format(len(x1_all), len(vocab)))

    ds = tf.data.Dataset.from_tensor_slices((x1_all, tok_all, dxdy_all))
    ds = ds.shuffle(buffer_size=len(x1_all))
    ds = ds.repeat()
    ds = ds.batch(settings['batch_size'])
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def run_trajectory(model, word, dx_dy, n_pts=2000):
    """
    Integrate n_pts Gaussian samples from t=0 → 1, returning snapshots.

    word:  str  (≤ MAX_CHARS chars)
    dx_dy: (2,) spatial offset
    Returns list of (n_pts, 2) numpy arrays, one per time slice.
    """
    ode_steps = settings['ode_steps']
    n_slices  = settings['plot_t_steps']
    dt        = 1.0 / ode_steps
    snap_every = max(1, ode_steps // (n_slices - 1))

    tokens  = np.array([encode_word(word)] * n_pts, dtype=np.int32)  # (n, 5)
    dxdy_np = np.tile(dx_dy, (n_pts, 1)).astype(np.float32)          # (n, 2)

    x       = tf.constant(np.random.randn(n_pts, 2).astype(np.float32))
    tokens_tf = tf.constant(tokens)
    dxdy_tf   = tf.constant(dxdy_np)

    snaps = [x.numpy()]
    for step in range(ode_steps):
        t = tf.fill((n_pts, 1), float(step) * dt)
        x = x + model(x, t, tokens_tf, dxdy_tf, training=False) * dt
        if (step + 1) % snap_every == 0 or step == ode_steps - 1:
            snaps.append(x.numpy())

    return snaps[:n_slices]


def plot_trajectory(model, save_path=None, step=None):
    """
    Plot ODE trajectories for each word in PLOT_WORDS.
    Each word gets one row; columns are time slices t=0…1.
    """
    lim      = settings['plot_axis_limit']
    n_slices = settings['plot_t_steps']
    n_words  = len(PLOT_WORDS)

    tf.random.set_seed(42)
    np.random.seed(42)

    fig, axes = plt.subplots(n_words, n_slices,
                             figsize=(3.5 * n_slices, 3 * n_words))
    if n_words == 1:
        axes = axes[np.newaxis, :]

    t_vals = np.linspace(0.0, 1.0, n_slices)

    for row, word in enumerate(PLOT_WORDS):
        snaps = run_trajectory(model, word, dx_dy=[0.0, 0.0])
        for col, (snap, t_val) in enumerate(zip(snaps, t_vals)):
            ax = axes[row, col]
            ax.scatter(snap[:, 0], snap[:, 1], s=1, alpha=0.4, color='steelblue')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title('t={:.2f}'.format(t_val), fontsize=9)
            if col == 0:
                ax.set_ylabel(word, fontsize=10, fontweight='bold')

    title = 'Flow Matching (Transformer)  |  d={} h={} L={}  |  ode_steps={}'.format(
        settings['d_model'], settings['n_heads'], settings['n_layers'],
        settings['ode_steps'])
    if step is not None:
        title = 'step {}  |  '.format(step) + title
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=90)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def make_checkpoint_manager(model, optimizer, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = settings.get('_checkpoint_dir', 'checkpoints_fm_text')
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64,
                              name='global_step')
    ckpt    = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                  global_step=global_step)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    return ckpt, manager, global_step


def restore_if_available(ckpt, manager):
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print("Restored: {}".format(manager.latest_checkpoint))
        return True
    print("No checkpoint found, starting from scratch.")
    return False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, ds, optimizer):
    print_period = settings['print_period']
    plot_period  = settings['plot_period']
    total_iters  = int(settings['train_iters'])

    progress_dir = settings.get('_progress_dir', 'training_progress_fm_text')
    os.makedirs(progress_dir, exist_ok=True)
    ckpt, manager, global_step = make_checkpoint_manager(model, optimizer)
    restore_if_available(ckpt, manager)

    start_step = int(global_step.numpy())
    if start_step >= total_iters:
        print("Already trained {} steps.".format(start_step))
        return float('nan')

    print("Resuming from step {}.".format(start_step))
    start = time()
    itr   = iter(ds)
    loss  = None

    for i in range(start_step, total_iters + 1):
        loss = model.train_step(next(itr), optimizer)

        if i % print_period == 0:
            global_step.assign(i)
            loss_val = loss.numpy()
            print("{} loss: {:.6f}, {:.1f}s".format(i, loss_val, time() - start))
            if np.isnan(loss_val):
                print("NaN loss — stopping.")
                break

        if i % plot_period == 0:
            manager.save()
            plot_trajectory(
                model,
                save_path='{}/step_{:07d}.png'.format(progress_dir, i),
                step=i)

    return loss.numpy() if loss is not None else float('nan')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_settings():
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU: {}".format(gpus[0].name) if gpus else "WARNING: No GPU detected")
    print("Using settings:")
    for k, v in settings.items():
        print('  {}: {}'.format(k, v))
    print("Vocab size: {}  |  Max chars: {}".format(VOCAB_SIZE, MAX_CHARS))
    print("Training words: {}  |  Plot words: {}".format(
        settings['n_train_words'], PLOT_WORDS))


def train_and_run_model(display=True, mode='full'):
    global PLOT_WORDS
    if mode == 'single':
        settings.update(settings['_single_letter'])
        settings['_checkpoint_dir'] = 'checkpoints_fm_single'
        settings['_progress_dir']   = 'training_progress_fm_single'
        PLOT_WORDS = PLOT_WORDS_SINGLE
        print("*** Single-letter PoC mode ***")
    elif mode == 'two':
        settings.update(settings['_two_letter'])
        settings['_checkpoint_dir'] = 'checkpoints_fm_two'
        settings['_progress_dir']   = 'training_progress_fm_two'
        PLOT_WORDS = PLOT_WORDS_TWO
        print("*** Two-letter mode ***")
    else:
        settings.setdefault('_checkpoint_dir', 'checkpoints_fm_text')
        settings.setdefault('_progress_dir',   'training_progress_fm_text')

    print_settings()
    ds = create_dataset(mode=mode)

    model = TextVelocityField()
    # Warmup to build weights before summary/checkpoint
    dummy_tokens = tf.zeros((1, MAX_CHARS), dtype=tf.int32)
    dummy_dxdy   = tf.zeros((1, 2))
    model(tf.zeros((1, 2)), tf.zeros((1, 1)), dummy_tokens, dummy_dxdy)
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
    parser = argparse.ArgumentParser(
        description='Flow Matching Transformer on 2D text point clouds.')
    parser.add_argument('--start-new', action='store_true',
                        help='Delete existing checkpoints and training images.')
    parser.add_argument('--single-letter', action='store_true',
                        help='Single-letter PoC: d_model=128, 2 layers, A–Z vocab.')
    parser.add_argument('--two-letter', action='store_true',
                        help='Two-letter mode: d_model=128, 3 layers, random 2-char pairs.')
    args = parser.parse_args()

    if args.single_letter:
        mode = 'single'
    elif args.two_letter:
        mode = 'two'
    else:
        mode = 'full'

    dir_map = {
        'single': ('checkpoints_fm_single', 'training_progress_fm_single'),
        'two':    ('checkpoints_fm_two',    'training_progress_fm_two'),
        'full':   ('checkpoints_fm_text',   'training_progress_fm_text'),
    }
    ckpt_dir, progress_dir = dir_map[mode]

    if args.start_new:
        for path in (ckpt_dir, progress_dir):
            if os.path.exists(path):
                shutil.rmtree(path)
                print("Deleted: {}".format(path))

    train_and_run_model(mode=mode)
