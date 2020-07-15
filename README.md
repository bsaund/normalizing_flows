# Normalizing Flows

Read the [accompianing blog post](https://www.bradsaund.com/post/normalizing_flows/)

Normalizing flows have become popular for modeling distributions of data, for example unsupervised learning of image datasets. 

For my first foray into Normalizing Flows I followed this [great tutorial](https://blog.evjang.com/2018/01/nf2.html), which was originally written in Tensorflow 1. This repo is my implementation of the modern normalizing flow examples from the tutorial in tensorflow 2.


## Installation
1. Install [tensorflow](https://www.tensorflow.org/install), [tensorflow_probability](https://www.tensorflow.org/probability), [matplotlib](https://matplotlib.org/users/installing.html), and Pillow. Clone this repo.
2. Train and run the model `./normalizing_flows.py`. It'll take about an hour to train. Reduce `train_iters` in in the `settings` dict to reduce the training time

You should first see the training points: 

<img src="https://github.com/bsaund/normalizing_flows/blob/master/pictures/BRAD_samples.png" width="70">

After training you should see the learned mapping from a normal distribution to the training samples

<img src="https://github.com/bsaund/normalizing_flows/blob/master/pictures/RealNVP_BRAD_all_layer.png" width="600">




## What is going on?
`normalizing_flows.py` implements RealNVP. The [image](pictures/RealNVP_BRAD_all_layer.png) shows the transformation of samples from a 2D Gaussian through each of the layers of the network. The quadrants of the Gaussian are color-coded to visualize how the gaussian transforms.

It is composed 8 repeated units of [RealNVP layers](https://github.com/bsaund/normalizing_flows/blob/15f40e26ae2dd7f3646031642a08765546b9ddb9/normalizing_flows.py#L75-L78) and [Permutations](https://github.com/bsaund/normalizing_flows/blob/15f40e26ae2dd7f3646031642a08765546b9ddb9/normalizing_flows.py#L83), with occasional batch normaliziation bijectors. 

Honestly, `tensorflow_distributions` does all of the heavy-lifting by implementing `RealNVP` layers, which take care of splitting the data and (most importantly) computing the jacobian needed to compute the gradient.
```
        for i in range(settings['num_bijectors']):
            self.bijector_fns.append(tfp.bijectors.real_nvp_default_template(hidden_layers=[512,512]))
            bijectors.append(
                tfb.RealNVP(num_masked=self.num_masked,
                            shift_and_log_scale_fn=self.bijector_fns[-1])
            )
            if i%2 == 0:
                bijectors.append(tfb.BatchNormalization())
            bijectors.append(tfb.Permute(permutation=[1,0]))
```

## What to do next?
- Try making your own pictures. I just wrote "BRAD" in google sheets and exported it as `.png`.
- Try changing the number of layers, types of layers, or other [settings](https://github.com/bsaund/normalizing_flows/blob/ff515f1c3b4c97c646ff375fa6de4f4a7847b256/normalizing_flows.py#L19-L25)
- Implement a normalizing flow on some image training data


## Citation:
Feel free to copy and use this code in your projects. If you would like to cite this repo: 

```
@misc{bsaund_2020_flow,
    author = {Brad Saund},
    title = {Normalizing Flows},
    year = 2020,
    month = {april},
    publisher = {Github},
    journal = {GitHub repository},
    url = {https://github.com/bsaund/normalizing_flows}
}
```

