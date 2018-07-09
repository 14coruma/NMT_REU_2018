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

def buildImages(files, targets, type):
    images = []
    for file in pb.progressbar(files):
        targets.append(file)
        with open(file, "rb") as f:
            if type == "Byte":
                images.append(bytePlot(list(f.read())))
            else:
                print("Invalid Type")
            makeimage(file, images[-1])
    return images, targets

def bytePlot(f):
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
    pageNames = []
    pageSize = 1000
    pages = range(math.ceil(len(files)/pageSize))
    for page in pages:
        print("\nPage {}/{}".format(page+1, len(pages)))
        images = []
        gc.collect() # Garbage collect
        start = page*pageSize
        for item in pb.progressbar(files[start:start+pageSize]):
            targets.append(item)
            images.append(smp.imread(item))
        pageNames.append("./pages/images_page{}.npy".format(page))
        np.save(pageNames[-1], images)
    return targets, pageNames 

def create(files):
    options = ["Byte", "Markov"]
    type = options[int(ui.prompt("Choose a visualization type", options))]

    targets = []
    pageNames = []
    pageSize = 1000
    pages = range(math.ceil(len(files)/pageSize))
    for page in pages:
        print("\nPage {}/{}".format(page+1, len(pages)))
        gc.collect() # Garbage collect
        start = page*pageSize
        images, targets = buildImages(files[start:start+pageSize], targets, type)
        pageNames.append("./pages/images_page{}.npy".format(page))
        np.save(pageNames[-1], images)
    return targets, pageNames

def process(directory): 
    files = []

    options = ["Load", "Create"]
    choice = options[int(ui.prompt(options=options))]

    if choice == "Load":
        for item in os.listdir(directory):
            if( os.path.isfile(os.path.join(directory, item)) and
            item.endswith(".bmp") ):
                files.append(os.path.join(directory, item))
        targets, pageNames = load(files)

    elif choice == "Create":
        for item in os.listdir(directory):
            if( os.path.isfile(os.path.join(directory, item)) and
            (item.endswith(".pdf") or item.endswith(".file")) ):
                files.append(os.path.join(directory, item))
        targets, pageNames = create(files)

    else:
        quit()
    
    targets = [name.split('/')[-1][:5] for name in targets]
    return pageNames, targets
