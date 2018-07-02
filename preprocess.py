#!/usr/bin/python3
import ui
import scipy.misc as smp # Required to export original png
import numpy as np
import math # Aids "length" variable
import multiprocessing as mp # PARALELL PROCESSING
import os
import time
from tqdm import tqdm

def getdims(f):
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

def makearray(f):
    dimensions = getdims(f)
    data = np.array(f)
    data = np.pad(
        data, (0, dimensions[0]-(len(data)%dimensions[0])), 'constant')
    data = np.reshape(data, (-1, dimensions[0]))
    return data

def makeimage(name, f):
    img = smp.toimage(f)
    smp.imsave(name+"(original).bmp", img)

def load(files):
    targets = []
    images = []
    for item in files:
        targets.append(item)
        with open(item, "rb") as f:
            images.append( makearray(list(f.read())) )
    return targets, images

def create(files):
    targets = []
    images = []
    for item in files:
        targets.append(item)
        with open(item, "rb") as f:
            images.append( makearray(list(f.read())) )
            makeimage(item, images[-1])
    return targets, images

def process(directory): 
    files = []

    choice = int(ui.prompt("(1)Load (2)Create"))
    if choice == 1:
        for item in os.listdir(directory):
            if( os.path.isfile(os.path.join(directory, item)) and
            item.endswith(".bmp") ):
                files.append(os.path.join(directory, item))
        targets, images = load(files)

    elif choice == 2:
        for item in os.listdir(directory):
            if( os.path.isfile(os.path.join(directory, item)) and
            (item.endswith(".pdf") or item.endswith(".file")) ):
                files.append(os.path.join(directory, item))
        targets, images = create(files)

    else:
        quit()
    
    targets = [name.split('/')[-1][:5] for name in targets]
    return images, targets
