#!/usr/bin/python3

import preprocess as pproc
import sys
#import features as ft
#import trainer as tr
import ui

def main():
    images, targets = pproc.process(sys.argv[1:])
    """
    data = ft.features(images)

    tr.train(data, targets)
    """
if __name__ == "__main__":
    main()
