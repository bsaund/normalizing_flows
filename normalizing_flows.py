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
    'learning_rate': 1e-5,
    'train_iters': 2e5,
    'visualize_data': False,
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


def plot_layers(dist, final=False):
    """
    Generate samples from the base distribution and visualize the motion of the points after each 
    layer transformation
    """
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
    i = 0
    # for i in range(len(results)):
    for r in range(rows):
        for c in range(cols):
            if i >= len(results):
                break
            X1 = results[i].numpy()
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
            arr[r, c].set_xlim([-5, 5])
            arr[r, c].set_ylim([-5, 5])
            arr[r, c].set_title(names[i])
            arr[r, c].axis('equal')
            i += 1
    plt.show()

    if not final:
        return

    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
    plt.axis('equal')
    plt.show()


def train(model, ds, optimizer, print_period=1000):
    """
    Train `model` on dataset `ds` using optimizer `optimizer`,
    printing the current loss every `print_period` iterations.
    Loss tensor stays on GPU between prints to avoid CPU-GPU sync overhead.
    """
    start = time()
    itr = ds.__iter__()
    for i in range(int(settings['train_iters'] + 1)):
        X = next(itr)
        loss = model.train_step(X, optimizer)
        if i % print_period == 0:
            loss_val = loss.numpy()
            print("{} loss: {}, {}s".format(i, loss_val, time() - start))
            if np.isnan(loss_val):
                break
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
