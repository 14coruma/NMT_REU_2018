#!/usr/bin/python3
import numpy as np
import gc
import os
from multiprocessing import Pool

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
    kps = []
    try:
        kps = alg.detect(img, None)
    except:
        pass

    # Get first sorted <vector_size> points.
    kps = sorted(kps, key=lambda x: x.response)[:vector_size]
    dsc = np.zeros((2, 2))
    if len(kps) > 0: # Don't compute if no keypoints were found
        kps, dsc = alg.compute(img, kps)

    # Fill with zeros if no keypoints are found
    if len(kps) < vector_size:
        dsc = np.zeros(shape=(vector_size, descriptor_size))

    # Flatten and normalize descriptors
    dsc = dsc.flatten()
    dsc = np.divide(dsc, 256)

    return dsc

def extract_ORB(img):    
    img = img.astype(np.uint8)
    alg = cv2.ORB_create()
    vector_size = 32
    descriptor_size = 32
    return describe_keypoints(img, alg, vector_size, descriptor_size)

def extract_SIFT(img):
    img = img.astype(np.uint8)
    alg = cv2.xfeatures2d.SIFT_create()
    vector_size = 32
    descriptor_size = 128
    return describe_keypoints(img, alg, vector_size, descriptor_size)

def extract_LBP(img):
    img = img.astype(np.uint8)
    points = 32
    radius = 16
    lbp = local_binary_pattern(img, points, radius, method="uniform")
    hist = np.array(exposure.histogram(lbp, nbins=16)[0])
    hist = np.divide(hist, sum(hist)) # Normalize histogram
    return hist

def extract_Gabor(img):
    img = img.astype(np.uint8)
    kernels = create_gabor_kernels()
    float_img = img_as_float(img)
    feats = compute_feats(float_img, kernels).flatten()
    hist = exposure.histogram(float_img, nbins=16)
    return np.append(feats, hist)

def extract_Entropy(img):
    img = img.astype(np.uint8)
    img = entropy(img, disk(5))
    hist = exposure.histogram(img, nbins=16)[0]
    #hist = np.divide(hist, sum(hist)) # Normalize histogram
    return hist

def features(pageNames):
    """Loop through images, extracting desired features"""
    options = ["ORB", "SIFT", "LBP", "Gabor", "Entropy"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    switch = {
        0: extract_ORB,
        1: extract_SIFT,
        2: extract_LBP,
        3: extract_Gabor,
        4: extract_Entropy,
    }
    fn = switch.get(int(res))

    print("Should take less than {} minutes.".format(len(pageNames)*2))
    print("Please wait...\n")

    # Run this with a pool of 3 agents until finished
    data = []
    for pageName in pb.progressbar(pageNames):
        gc.collect()
        images = np.load(pageName)
        with Pool(processes=3) as pool:
            if len(data) == 0:
                data = pool.map(fn, images, 16)
            else:
                data = np.concatenate((data, pool.map(fn, images, 16)))
        # Remove paged files (to clear up disk space)
        os.unlink(pageName)

    return data 
