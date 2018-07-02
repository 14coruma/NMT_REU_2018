#!/usr/bin/python3

import preprocess as pproc
import features as ft
import trainer as tr

def main():
    images, targets = pproc.process()

    data = ft.features(images)

    tr.train(data, targets)

if __name__ == "__main__":
    main()
