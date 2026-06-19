#!/usr/bin/env python
from __future__ import print_function

import os
# Suppress the wrapt C-extension warning that fires on Python 3.12
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "1")

import tensorflow as tf
import tensorflow_probability as tfp

# TFP requires Keras 2 (tf_keras), not Keras 3 bundled with TF 2.16+.
# tf_keras is installed automatically by `tensorflow-probability[tf]`.
import tf_keras

import matplotlib.pyplot as plt
import numpy as np
from generate_points import create_uniform_points, create_points, visualize_data
from time import time

tfd = tfp.distributions
tfb = tfp.bijectors

# Allow TF to grow GPU memory incrementally rather than claiming it all at once.
# Important on a shared GPU (e.g. one also driving the display).
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)


def make_nvp_network(hidden_units, output_units):
    """Return a Keras model used as the shift-and-log-scale network for RealNVP."""
    return tf_keras.Sequential([
        tf_keras.layers.Dense(h, activation="relu") for h in hidden_units
    ] + [tf_keras.layers.Dense(output_units * 2)])


def nvp_shift_and_log_scale_fn(hidden_units, output_units):
    """
    Factory that returns a (callable, network) pair for tfb.RealNVP's
    shift_and_log_scale_fn.  Each call creates a fresh set of Keras weights,
    so call this once per bijector layer.  The returned network must be stored
    on the model so its variables are tracked for gradient updates.
    """
    net = make_nvp_network(hidden_units, output_units)

    def fn(x, input_depth=None, **kwargs):
        out = net(x)
        shift, log_scale = tf.split(out, 2, axis=-1)
        return shift, log_scale

    return fn, net

settings = {
    'batch_size': 1500,
    'method': 'NVP',
    'num_bijectors': 8,
    'learning_rate': 1e-4,
    'train_iters': 2e5,
    'visualize_data': False,
    'print_period': 1000,
    'plot_period': 500,
    'plot_axis_limit': 4.0,  # fixed axis bounds for all frames so video is stable
}


class Flow(tf_keras.Model):
    def __init__(self, **kwargs):
        super(Flow, self).__init__(**kwargs)
        self.flow = None
        # Keras networks backing the bijectors — tracked for trainable_variables
        self._bijector_nets = []

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    @property
    def trainable_variables(self):
        # Collect variables from all backing networks
        vars_ = []
        seen = set()
        for net in self._bijector_nets:
            for v in net.trainable_variables:
                if id(v) not in seen:
                    seen.add(id(v))
                    vars_.append(v)
        return vars_

    @tf.function
    def train_step(self, X, optimizer):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class MAF(Flow):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(MAF, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked

        bijectors = []
        for i in range(settings['num_bijectors']):
            # AutoregressiveNetwork is the modern, Keras-native replacement for
            # masked_autoregressive_default_template
            net = tfb.AutoregressiveNetwork(params=2, hidden_units=[512, 512], activation="relu")
            self._bijector_nets.append(net)
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=net)
            )
            bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0]),
            bijector=bijector)


class RealNVP(Flow):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(RealNVP, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked

        bijectors = []
        for i in range(settings['num_bijectors']):
            # Each layer needs its own network; store nets so their variables are tracked.
            # Note: Must store the bijectors separately, otherwise only a single set of
            # tf variables is created for all layers.
            fn, net = nvp_shift_and_log_scale_fn(hidden_units=[512, 512], output_units=num_masked)
            self._bijector_nets.append(net)
            bijectors.append(
                tfb.RealNVP(num_masked=self.num_masked, shift_and_log_scale_fn=fn)
            )

            bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0]),
            bijector=bijector)


def plot_layers(dist, final=False, save_path=None, step=None):
    """
    Generate samples from the base distribution and visualize the motion of the points after each
    layer transformation. If save_path is given, save to file instead of displaying.
    All subplots use fixed axis bounds (settings['plot_axis_limit']) so frames are stable in video.
    """
    lim = settings['plot_axis_limit']

    # Fixed seed so the same base points are used every frame — eliminates
    # random jitter between frames in the training video.
    tf.random.set_seed(42)
    x = dist.distribution.sample(8000)
    samples = [x]
    names = [dist.distribution.name]
    for bijector in reversed(dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)

    results = samples
    X0 = results[0].numpy()

    rows = 4
    cols = int(len(results) / rows) + (len(results) % rows > 0)

    f, arr = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    title = 'Step {:,}'.format(step) if step is not None else ''
    f.suptitle(title, fontsize=16, fontweight='bold')

    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = arr[r, c]
            if i >= len(results):
                ax.axis('off')
                continue
            X1 = results[i].numpy()
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_aspect('equal')
            ax.set_title(names[i])
            i += 1

    plt.tight_layout()
    if save_path:
        f.savefig(save_path, dpi=100)
        plt.close(f)
    else:
        plt.show()

    if not final:
        return

    fig2, ax2 = plt.subplots()
    X1 = results[-1].numpy()
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    ax2.scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    ax2.scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    ax2.scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    ax2.scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
    ax2.set_xlim([-lim, lim])
    ax2.set_ylim([-lim, lim])
    ax2.set_aspect('equal')
    if step is not None:
        ax2.set_title('Step {:,}'.format(step))
    if save_path:
        final_path = save_path.replace('.png', '_final.png')
        fig2.savefig(final_path, dpi=100)
        plt.close(fig2)
    else:
        plt.show()


def make_checkpoint_manager(model, optimizer, checkpoint_dir='checkpoints'):
    """
    Create a tf.train.Checkpoint + CheckpointManager for the model's networks,
    optimizer state, and a global step counter. Keeps the 3 most recent checkpoints.
    """
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
    ckpt_kwargs = {f'net_{i}': net for i, net in enumerate(model._bijector_nets)}
    ckpt_kwargs['optimizer'] = optimizer
    ckpt_kwargs['global_step'] = global_step
    ckpt = tf.train.Checkpoint(**ckpt_kwargs)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    return ckpt, manager, global_step


def restore_if_available(ckpt, manager):
    """Restore the latest checkpoint if one exists. Returns True if restored."""
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print("Restored checkpoint: {}".format(manager.latest_checkpoint))
        return True
    print("No checkpoint found, starting from scratch.")
    return False


def train(model, ds, optimizer):
    """
    Train `model` on dataset `ds` using optimizer `optimizer`.
    - Prints loss every settings['print_period'] steps.
    - Saves a layer-visualization PNG and checkpoint every settings['plot_period'] steps.
    - Uses a persistent global_step so filenames and logs are continuous across restarts.
    Loss tensor stays on GPU between syncs to avoid CPU-GPU sync overhead.
    """
    print_period = settings['print_period']
    plot_period  = settings['plot_period']
    total_iters  = int(settings['train_iters'])

    os.makedirs('training_progress', exist_ok=True)
    ckpt, manager, global_step = make_checkpoint_manager(model, optimizer)
    restore_if_available(ckpt, manager)

    start_step = int(global_step.numpy())
    if start_step >= total_iters:
        print("Already trained for {} steps, nothing to do.".format(start_step))
        return None

    print("Resuming from step {}.".format(start_step))
    start = time()
    itr = ds.__iter__()
    loss = None

    for i in range(start_step, total_iters + 1):
        X = next(itr)
        loss = model.train_step(X, optimizer)
        global_step.assign(i)

        if i % print_period == 0:
            loss_val = loss.numpy()
            print("{} loss: {}, {:.1f}s".format(i, loss_val, time() - start))
            if np.isnan(loss_val):
                break

        if i % plot_period == 0:
            manager.save()
            save_path = 'training_progress/step_{:07d}.png'.format(i)
            plot_layers(model.flow, save_path=save_path, step=i)

    return loss.numpy()


def print_settings():
    """
    display the settings used when creating the model
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU: {}".format(gpus[0].name))
    else:
        print("WARNING: No GPU detected, training on CPU")
    print("Using settings:")
    for k in settings.keys():
        print('{}: {}'.format(k, settings[k]))


def build_model(model):
    """
    Run a pass of the model to initialize the tensorflow network
    """
    x = model.flow.distribution.sample(8000)
    for bijector in reversed(model.flow.bijector.bijectors):
        x = bijector.forward(x)


def create_dataset():
    # pts = create_uniform_points(1000)
    # pts = create_points('two_moons.png', 10000)
    pts = create_points('BRAD.png', 10000)

    if settings['visualize_data']:
        visualize_data(pts)

    ds = tf.data.Dataset.from_tensor_slices(pts)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=9000)
    ds = ds.prefetch(3 * settings['batch_size'])
    ds = ds.batch(settings['batch_size'])

    return ds, pts


def train_and_run_model(display=True):
    print_settings()

    ds, pts = create_dataset()

    if settings['method'] == 'MAF':
        model = MAF(output_dim=2, num_masked=1)
    elif settings['method'] == 'NVP':
        model = RealNVP(output_dim=2, num_masked=1)

    model(pts)
    build_model(model)
    if display:
        model.summary()

    optimizer = tf_keras.optimizers.Adam(
        learning_rate=settings['learning_rate'], jit_compile=False)
    loss = train(model, ds, optimizer)

    if display:
        XF = model.flow.sample(2000)
        plot_layers(model.flow, final=True)

    return loss


def run_statistics_trial():
    """
    Runs 10 trials and reports the number of times training fails
    """
    final_loss = []
    for i in range(10):
        print()
        final_loss.append(train_model(display=False))
        print("Final loss for trial {} is {}".format(i, final_loss[-1]))

    print("Training failed {} of the time".format(np.sum(np.isnan(final_loss)) * 1.0 / len(final_loss)))


if __name__ == "__main__":
    train_and_run_model()
    # run_statistics_trial()
