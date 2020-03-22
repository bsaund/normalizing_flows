#!/usr/bin/env python
from __future__ import print_function


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import matplotlib.pyplot as plt
import numpy as np
from generate_points import create_points, visualize_data

import IPython


base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2]))


def make_nvp_layer():
    return tfb.RealNVP(
        num_masked=1,
        shift_and_log_scale_fn=tfb.real_nvp_default_template(
            hidden_layers=[512,512]))


def make_nvp_flow():
    num_bijectors = 4
    bijectors = []
    for i in range(num_bijectors):
        bijectors.append(make_nvp_layer())
        if i % 2 == 0:
            bijectors.append(tfb.BatchNormalization())
        bijectors.append(tfb.Permute(permutation=[1,0]))

    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=flow_bijector)
    return dist

def make_maf_flow():
    num_bijectors=8
    bijectors = []
    for i in range(num_bijectors):
        bijectors.append(tfb.MaskedAutoregressiveFlow(
            # shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            # hidden_layers=[512,512])))
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[512,512])
        )
        )
                         
        if i % 2 == 0:
            bijectors.append(tfb.BatchNormalization())
        bijectors.append(tfb.Permute(permutation=[1,0]))
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=flow_bijector)
    return dist


def visualize(dist, final=False):
    x = base_dist.sample(8000)
    samples = [x]
    names = [base_dist.name]
    for bijector in reversed(dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)

    results = samples

    X0 = results[0].numpy()
    f, arr = plt.subplots(1, len(results), figsize=(4*len(results), 4))
    for i in range(len(results)):
        X1 = results[i].numpy()
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
        arr[i].set_xlim([-10, 10])
        arr[i].set_ylim([-10, 10])
        arr[i].set_title(names[i])
        arr[i].axis('equal')
    plt.show()

    if not final:
        return
    
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
    plt.axis('equal')
    plt.show()


@tf.function
def train_step(dist, opt, x_samples):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(dist.log_prob(x_samples))
        variables = dist.trainable_variables
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(list(zip(gradients, variables)))
    return loss


def train(dist, ds, opt):
    print("Training")
    num_steps = int(2e4)
    itr = ds.__iter__()
    losses = []
    
    visualize(dist)
    for i in range(num_steps):
        x_samples = next(itr)
        loss = train_step(dist, opt, x_samples)
        
        if i%1000 == 0:
            print("{}/{}: loss={}".format(i, num_steps, loss))
            losses.append(loss.numpy())

    return losses


def prepare():
    pts = create_points('two_moons.png', 10000)
    visualize_data(pts)
    ds = tf.data.Dataset.from_tensor_slices(pts)
    ds = ds.batch(900)
    ds = ds.repeat()

    dist = make_nvp_flow()
    dist = make_maf_flow()
    opt = tf.keras.optimizers.Adam(1e-4)

    return dist, ds, opt



if __name__ == "__main__":
    dist, ds, opt = prepare()
    visualize(dist)
    # IPython.embed()
    losses = train(dist, ds, opt)
    visualize(dist, final=True)
    IPython.embed()
