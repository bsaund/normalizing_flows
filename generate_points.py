#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import IPython



def create_uniform_points(num_points):
    return np.array(np.random.uniform(-1, 1, (num_points,2)), dtype='float32')

def create_points(fn, num_points):
    with Image.open(fn) as image:
        w, h = image.size
        pts = []
        while len(pts) < num_points:
            pt = np.random.rand(2).astype('f')
            x = int((pt[0])*w)
            y = int((1-pt[1])*h)

            pxl = image.getpixel((x,y))
            if pxl[0] != 255:
                pts.append(pt*2-1)
    pts = np.array(pts)
    return pts

def visualize_data(pts):
    plt.scatter(pts[:,0], pts[:,1], s=5)
    plt.axis('equal')
    plt.show()
    


if __name__ == "__main__":
    print('hi')
    pts = create_points('two_moons.png', 10000)
    visualize_data(pts)
