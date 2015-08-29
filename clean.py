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
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion

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
    # Dialate the image first so that letters connected to large stains go away
    clusters = label(img, connectivity=1)

    props = regionprops(clusters)
    diameters = np.array([prop.equivalent_diameter for prop in props])
    euler_numbers = np.array([prop.euler_number for prop in props])
    eccentricities = np.array([prop.eccentricity for prop in props])
    areas = np.array([prop.filled_area for prop in props])
    cluster_sizes = np.bincount(clusters.ravel())
    size_threshold = threshold_otsu(cluster_sizes)
    print(size_threshold)
    is_big_enough = cluster_sizes > size_threshold
    out = np.copy(img)
    out[~is_big_enough[clusters]] = 255
    return out




    # Calculate mean width and height of clusters

# We can characterise the blobs on the basis of eccentricty, area and diameter
# We have the output images so we can learn the values from that data

def local_otsu_denoising(img):
    N = img.shape[0]
    radius = np.arange(10, N/4, 10)
    stds = []
    out_images = []
    for radii in radius:
        selem = square(radii)

        out = img.copy()
        # Improve the contrast of the image

        local_otsu = rank.otsu(img, selem)
        stds.append(out[out <= local_otsu].std())
        out[out >= local_otsu] = 255
        out_images.append(out)
    stds = np.array(stds)
    arg = stds.argmin()
    out = out_images[arg]

    return out


# After local_otsu_denoising you can find out the size of the letter then move
# the a window of that size over the image and apply otsu there.
# the size of the letter can median diameter of the bounding box
# Or maybe we can do otsu again on large blobs

# There are two cases the stain contains a letter or it does not contain a
# letter

def main(img):
    denoised = local_otsu_denoising(img)

    bin_denoised = np.copy(denoised)
    bin_denoised[bin_denoised < 255] = 0

    label_img = label(bin_denoised, connectivity=1)
    props = np.array(regionprops(label_img))
    diameters = np.array([prop.equivalent_diameter for prop in props])
    euler_numbers = np.array([prop.euler_number for prop in props])
    eccentricities = np.array([prop.eccentricity for prop in props])
    areas = np.array([prop.filled_area for prop in props])

    # XXX: 4 is a magic number
    stain_labels = [prop.label for prop in props[areas > 4*np.median(areas)]]
    stains = []
    stain_threshold = []
    for stain_label in stain_labels:
        stains.append(label_img == stain_label)
        stain_threshold.append(threshold_otsu(img[stains[-1]]))

    # Remove stain from denoised image
    # for stain, threshold in zip(stains, stain_threshold):
    unstained = np.copy(denoised)
    for stain, threshold in zip(stains, stain_threshold):
        unstained[np.logical_and(stain, denoised > threshold)] = 255
    plt.imshow(denoised, cmap='gray')
    plt.show()
    plt.imshow(unstained, cmap='gray')
    plt.show()


if __name__ == "__main__":
    images = get_data("./data/train/")

    img = images[9]
    denoised = local_otsu_denoising(img)

    bin_denoised = np.copy(denoised)
    bin_denoised[bin_denoised < 255] = 0

    bin_denoised = binary_erosion(bin_denoised)

    label_img = label(bin_denoised, connectivity=1)
    props = np.array(regionprops(label_img))
    diameters = np.array([prop.equivalent_diameter for prop in props])
    euler_numbers = np.array([prop.euler_number for prop in props])
    eccentricities = np.array([prop.eccentricity for prop in props])
    areas = np.array([prop.filled_area for prop in props])

    # XXX: 4 is a magic number
    stain_labels = [prop.label for prop in props[areas > 4*np.median(areas)]]
    stains = []
    stain_threshold = []
    for stain_label in stain_labels:
        stains.append(label_img == stain_label)
        stain_threshold.append(threshold_otsu(img[stains[-1]]))

    # Remove stain from denoised image
    # for stain, threshold in zip(stains, stain_threshold):
    unstained = np.copy(denoised)
    for stain, threshold in zip(stains, stain_threshold):
        unstained[np.logical_and(stain, denoised > threshold)] = 255
