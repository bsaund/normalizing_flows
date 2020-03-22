#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import IPython





def create_points(fn, num_points):
    with Image.open(fn) as image:
        w, h = image.size
        pts = []
        while len(pts) < num_points:
            pt = np.random.rand(2).astype('f')
            x = int(pt[0]*w)
            y = int(pt[1]*h)

            pxl = image.getpixel((x,y))
            if pxl[0] != 255:
                pts.append(pt)
    pts = np.array(pts)
    return pts


if __name__ == "__main__":
    print('hi')
    pts = create_points('two_moons.png', 10000)
    plt.scatter(pts[:,0], pts[:,1])
    plt.show()
