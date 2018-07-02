#!/usr/bin/python3
import numpy as np

import cv2 # (ORB, SIFT, cvtColor(grey))
from skimage import io # Load image from file

from skimage import exposure # For creating histogram
from skimage.feature import local_binary_pattern

import ui # ui.prompt() 
import progressbar as pb # Display progressbar

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
    options = ["ORB", "SIFT", "LBP", "Gabor", "Entropy", "LBP and Entropy"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    type = options[int(res)]

    # Load image for testing. TODO: Remove this line
    images.append(io.imread("../NMT_REU/data/small/CLEAN_06cofigfs.file.bmp"))

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
        else:
            print("ERROR: Type " + type + " not found (features.extract_features())\n")
            return 1

    return data
