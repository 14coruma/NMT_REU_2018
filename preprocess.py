#!/usr/bin/python3
import ui
import numpy as np
import scipy.misc as smp # Required to export original png import numpy as np
import math # Aids "length" variable import numpy as np
import multiprocessing as mp # PARALELL PROCESSING import os
import os
import time
from tqdm import tqdm
import progressbar as pb

import gc

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
        images.append(smp.imread(item))
    return targets, images

def create(files):
    targets = []
    pageNames = []
    pageSize = 1000
    pages = range(math.ceil(len(files)/pageSize))
    print("Number of pages: {}".format(len(pages)))
    for page in pages:
        images = []
        gc.collect() # Garbage collect
        start = page*pageSize
        for item in pb.progressbar(files[start:start+pageSize]):
            targets.append(item)
            with open(item, "rb") as f:
                images.append( makearray(list(f.read())) )
                makeimage(item, images[-1])
        pageNames.append("./pages/images_page{}.npy".format(page))
        np.save(pageNames[-1], images)
    return targets, pageNames

def process(directory): 
    files = []

    options = ["Load", "Create"]
    choice = int(ui.prompt(options=options))
    if choice == 0:
        for item in os.listdir(directory):
            if( os.path.isfile(os.path.join(directory, item)) and
            item.endswith(".bmp") ):
                files.append(os.path.join(directory, item))
        targets, images = load(files)

    elif choice == 1:
        for item in os.listdir(directory):
            if( os.path.isfile(os.path.join(directory, item)) and
            (item.endswith(".pdf") or item.endswith(".file")) ):
                files.append(os.path.join(directory, item))
        targets, pageNames = create(files)

    else:
        quit()
    
    targets = [name.split('/')[-1][:5] for name in targets]
    return pageNames, targets
