#!/usr/bin/python3
import numpy as np

import cv2 # (ORB, SIFT, cvtColor(grey))
from scipy import ndimage as nd # For convolving kernel
from skimage import exposure # For creating histogram
from skimage.util import img_as_float # Needed for gabor filter
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk # Create a disk around a pixel (for entropy)

import ui # ui.prompt() 
import progressbar as pb # Display progressbar

def create_gabor_kernels(n_theta=4):
    """Generate gabor kernles"""
    kernels = []
    for theta in range(n_theta):
        theta = theta / 8. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

def compute_feats(image, kernels):
    """Create feature vector from gabor kernels and image"""
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def describe_keypoints(img, alg, vector_size, descriptor_size, display=False):
    """Create description vector for keypoints in an image"""
    # Finding image keypoints
    kps = alg.detect(img, None)

    # Get first sorted <vector_size> points.
    kps = sorted(kps, key=lambda x: x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)

    # Fill with zeros if no keypoints are found
    if len(kps) < vector_size:
        dsc = np.zeros(shape=(vector_size, descriptor_size))

    # Flatten and normalize descriptors
    dsc = dsc.flatten()
    dsc = np.divide(dsc, 256)

    return dsc

def features(images):
    options = ["ORB", "SIFT", "LBP", "Gabor", "Entropy"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    type = options[int(res)]

    data = []
    for img in pb.progressbar(images): # Process each image
        if type == "ORB":              # Corner features
            alg = cv2.ORB_create()
            vector_size = 32
            descriptor_size = 32
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size))
        elif type == "SIFT":           # Corner features (patented)
            alg = cv2.xfeatures2d.SIFT_create()
            vector_size = 32
            descriptor_size = 32
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size))
        elif type == "LBP":
            points = 32
            radius = 16
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(grey, points, radius, method="uniform")
            hist = np.array(exposure.histogram(lbp, nbins=16)[0])
            hist = np.divide(hist, sum(hist)) # Normalize histogram
            data.append(hist)
        elif type == "Gabor":
            # prepare filter bank kernels
            kernels = create_gabor_kernels()
            float_img = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            feats = compute_feats(float_img, kernels).flatten()
            hist = exposure.histogram(float_img, nbins=16)
            data.append(np.append(feats, hist))
        elif type == "Entropy":
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grey = entropy(grey, disk(5))
            hist = exposure.histogram(grey, nbins=16)[0]
            hist = np.divide(hist, sum(hist)) # Normalize histogram
            data.append(hist)
        else:
            print("ERROR: Invalid feature extraction method")
            return 1

    return data
