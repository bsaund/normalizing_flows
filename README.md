# Normalizing Flows

Normalizing flows have become popular for modeling distributions of data, for example unsupervised learning of image datasets. 

For my first foray into Normalizing Flows I followed this great tutorial: https://blog.evjang.com/2018/01/nf2.html, which was originally written in Tensorflow 1. This repo is my implementation of the modern normalizing flow examples from the tutorial in tensorflow 2.


## Installation
1. Install [tensorflow](https://www.tensorflow.org/install), [tensorflow_probability](https://www.tensorflow.org/probability), [matplotlib](https://matplotlib.org/users/installing.html), and IPython (optional). Clone this repo.
2. Train and run the model `./normalizing_flows.py`. It'll take about an hour to train. 

You should first see the training points: 

<img src="https://github.com/bsaund/normalizing_flows/blob/master/pictures/BRAD_samples.png" width="70">

After training you should see the learned mapping from a normal distribution to the training samples

<img src="https://github.com/bsaund/normalizing_flows/blob/master/pictures/RealNVP_BRAD_all_layer.png" width="600">




## What is going on?
`normalizing_flows.py` implements RealNVP. 


