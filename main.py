#!/usr/bin/python3

import sys

import preprocess as pproc
import features as ft
import trainer as tr
import ui

def main():
    # Create images & extract features (until user quits)
    doneExtracting = False
    while not doneExtracting:
        pageNames, targets = pproc.process(sys.argv[1])
        data = ft.features(pageNames)

        # Create and evaluate model (until user quits)
        doneTraining = False
        while not doneTraining:
            tr.train(data, targets)
            
            options = ["Try another model", "Extract new features", "Quit"]
            res = options[int(ui.prompt(options=options))]
            if res == "Quit":
                doneTraining = True
                doneExtracting = True
            elif res == "Extract new features":
                doneTraining = True

if __name__ == "__main__":
    main()
