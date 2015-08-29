# Given an image of NxN
# Vary the window size from 5 to N/2 and set the pixels
# find otsu threshold and set all the pixels less than the threshold to zero
# and find the variance of the image
# Take the image with the lowest variance
from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt

from skimage.morphology import square
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.measure import label

import os
import numpy as np


def get_data(image_dir):
    file_names = [file_name for file_name in os.listdir(image_dir)
                  if file_name.endswith(".png")]
    images = []
    for file_name in file_names:
        images.append(imread(image_dir + file_name, as_grey=True))
    images = np.array(images)
    return images



# Naive local otsu doesn't help much, for smaller radii there is a large amount
# of salt and pepper type of noise, it appears that there also large connected
# components

# Letters are connected components with pretty much uniform I classify that
# something is a letter or not by taking the mean width and mean length, doing
# clustring on it and remove the things which don't appear as letter.
# Things which don't appear as letter can be:
# XXX: This is assuming printed letter and non cursive writing
# Connected components which are too small
# with too small or too large length
# with too small or too large width


def remove_small_components(img):
    # This method works well, just that the dot on i is going away
    clusters = label(img, connectivity=2)
    cluster_sizes = np.bincount(clusters.ravel())
    size_threshold = threshold_otsu(cluster_sizes)
    print(size_threshold)
    is_big_enough = cluster_sizes > size_threshold
    print(np.bincount(is_big_enough))
    out = np.copy(img)
    out[~is_big_enough[clusters]] = 255
    return out

    # Calculate mean width and height of clusters


if __name__ == "__main__":
    images = get_data("./data/train/")
    in_images = images[0]

    out_images = []
    radius = [10, 20, 30, 40, 50]
    for radii in radius:
        selem = square(radii)

        local_otsu = rank.otsu(in_images, selem)
        out = in_images.copy()
        out[out <= local_otsu] = 0
        out_images.append(out)
    img = out_images[1]
    out = remove_small_components(img)
