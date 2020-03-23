#!/usr/bin/env python
from __future__ import print_function


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import matplotlib.pyplot as plt
import numpy as np
from generate_points import *
from time import time

import IPython


class MAF(tf.keras.models.Model):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(MAF, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked
        self.shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[128,128])

        num_bijectors = 8
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.shift_and_log_scale_fn))
            bijectors.append(tfb.Permute(permutation=[1,0]))
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0]),
            bijector=bijector)
        
    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    def getFlow(self, num):
        return self.flow.sample(num)
    

@tf.function
def train_step(model, X, optimizer):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(model.flow.log_prob(X))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def visualize(dist, final=False):
    # IPython.embed()
    # x = base_dist.sample(8000)
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
    cols = int(len(results)/rows)

    f, arr = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    i = 0
    # for i in range(len(results)):
    for r in range(rows):
        for c in range(cols):
            X1 = results[i].numpy()
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            arr[r,c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            arr[r,c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            arr[r,c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
            arr[r,c].set_xlim([-5, 5])
            arr[r,c].set_ylim([-5, 5])
            arr[r,c].set_title(names[i])
            arr[r,c].axis('equal')
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


def train(model, X, optimizer):
    start = time()
    for i in range(501):
        loss = train_step(model, X, optimizer)
        if i % 50 == 0:
            print("{} loss: {}, {}s".format(i, loss, time() - start))
            

def run_tf2_tutorial():
    pts = create_uniform_points(1000)
    # pts = create_points('two_moons.png', 1000)
    # visualize_data(pts)

    model = MAF(output_dim=2, num_masked=1)
    model(pts)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train(model, pts, optimizer)


    XF = model.flow.sample(10000)
    visualize(model.flow)
    # visualize_data(XF.numpy())


if __name__ == "__main__":
    run_tf2_tutorial()
