#!/usr/bin/python3

import ui
import numpy as np
import math # Aids "length" variable
import multiprocessing as mp # PARALELL PROCESSING

def getdims(f):
    """ Processes image with, height, and size.
    @param f is the file.
    @return (width, height, filesize(bytes).
    """
    size = len(f)
    # width
    # TODO: make this prettier
    if (size <= 10):
        width = 32
    elif (10 < size <= 30):
        width = 64
    elif (30 < size <= 60):
        width = 128
    elif (60 < size <= 100):
        width = 256
    elif (100 < size <= 200):
        width = 384
    elif (200 < size <= 500):
        width = 512
    elif (500 < size <= 1000):
        width = 768
    else:
        width = 1024

    # (width, height) 
    return (width, math.ceil(size*1000 // width)+ 1 )


def makearray(f):
    dimensions = getdims(f)
    
    # Creating Image-----------------------------------------------------------
    data = np.array(f)
    data = np.pad(
        data, (0, dimensions[0]-(len(data)%dimensions[0])), 'constant')
    data = np.reshape(data, (-1, dimensions[0]))

    return data

def process(files): 
    targets = []
    images = []

    # User input whether to make images or not. Defaults to no
    choice = int(ui.prompt("(1)Load (2)Create"))
    if choice == 1:
        for name in files:
            if ".bmp" name[-3:]:
                with open(name, "rb") as f:
                    images.append( makearray(list(f.read())) )

    targets = [name.split('/')[-1][:5] for name in targets]
    return images, targets

