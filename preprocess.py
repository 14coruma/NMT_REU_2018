#!/usr/bin/python3
import numpy as np
import scipy.misc as smp # Save images
import scipy.ndimage as snd # Load images
import math # Aids "length" variable
import progressbar as pb
from math import isinf
import os
import gc
import ui

def getDims(f):
    """Returns (width, height) of byte plot image"""
    size = len(f) * .001
    if (size <= 10):
        width = 32
    elif (size <= 30):
        width = 64
    elif (size <= 60):
        width = 128
    elif (size <= 100):
        width = 256
    elif (size <= 200):
        width = 384
    elif (size <= 500):
        width = 512
    elif (size <= 1000):
        width = 768
    else:
        width = 1024
    return (width, math.ceil(size*1000 // width)+ 1)

def bytePlot(f):
    """Creates byte plot from byte array. Returns image array"""
    dimensions = getDims(f)
    data = np.array(f)
    data = np.pad(
        data, (0, dimensions[0]-(len(data)%dimensions[0])), 'constant')
    data = np.reshape(data, (-1, dimensions[0]))
    return data

def markovPlot(f):
    """Creates markov plot from byte array. Returns image array"""
    p = np.zeros(shape=(256, 256))
    for i in range(len(f)-1):
        row = f[i]
        col = f[i+1]
        p[row, col] += 1

    for row in range(256):
        sum = np.sum(p[row])
        if sum != 0:
            p[row] /= sum

    # Normalize
    p = (1 / np.ndarray.max(p)) * p
    p *= 255

    img = np.zeros(shape=(256, 256))
    for row in range(256):
        for col in range(256):
            val = p[row, col]
            val = val if not isinf(val) else 0
            img[row, col] = val
    
    return img.astype(np.uint8)

def getOrder(f):
    """Find order of hilbert curve needed for a file size"""
    order = 0
    while (2 ** order) ** 2 < len(f):
        order += 1
    return order

def rot(n, x, y, rx, ry):
    """Rotate/flip a hilbert quadrant"""
    if ry == 0:
        if rx == 1:
            x = (n-1) - x
            y = (n-1) - y
        t = x
        x = y
        y = t
    return x, y

def d2xy(n, d):
    """Convert a linear position, d, to hilbert (x,y)"""
    x, y, rx, ry, s, t = 0, 0, 0, 0, 0, d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t = t // 4
        s *= 2
    return x, y

def hilbertPlot(f):
    """Create hilbert plot, which is plot of bytes in shape of hilbert curve"""
    order = getOrder(f)
    img = np.zeros(shape=(2 ** order, 2 ** order))
    for d in range(len(f)):
        x, y = d2xy(2 ** order, d)
        img[x, y] = f[d]
    return img

def buildImages(files, targets, type):
    """Builds mages from array of filenames. Returns (images, targets)"""
    images = []
    for file in files:
        targets.append(file)
        with open(file, "rb") as f:
            if type == "Byte":
                images.append(bytePlot(list(f.read())))
            elif type == "Markov":
                images.append(markovPlot(list(f.read())))
            elif type == "Hilbert":
                images.append(hilbertPlot(list(f.read())))
            smp.imsave("{}.png".format(file), images[-1])
    return images, targets

def loadImages(files, targets):
    """Loads an array of png images. Returns (images, targets)"""
    images = []
    for file in files:
        targets.append(file)
        images.append(snd.imread(file))
    return images, targets

def imagePages(files, choice):
    """Pages images into npy file in groups of 100"""
    options = ["Byte", "Markov", "Hilbert"]
    type = options[int(ui.prompt("Choose a visualization type", options))]

    targets = []
    pageNames = []
    pageSize = 100
    pages = range(math.ceil(len(files)/pageSize))
    for page in pb.progressbar(pages):
        # print("\nPage {}/{}".format(page+1, len(pages)))
        gc.collect() # Garbage collect

        images = []
        start = page*pageSize
        if choice == "Create":
            images, targets = buildImages(files[start:start+pageSize], targets, type)
        elif choice == "Load":
            images, targets = loadImages(files[start:start+pageSize], targets)
        pageNames.append("./pages/images_page{}.npy".format(page))
        np.save(pageNames[-1], images)
    return targets, pageNames

def process(directory): 
    """Process each file in a directory, saving or loading images as directed"""
    files = []

    options = ["Load", "Create"]
    choice = options[int(ui.prompt(options=options))]

    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            filename = os.path.join(directory, item)
            if choice == "Load" and item.endswith(".png"):
                files.append(filename)
            elif choice == "Create" and item.endswith(".file"):
                files.append(filename)

    filenames, pageNames = imagePages(files, choice)
    
    targets = [name.split('/')[-1][:5] for name in filenames]
    return pageNames, targets, filenames
