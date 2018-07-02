#!/usr/bin/python3

import ui
import progressbar as pb

def features(images):
    options = ["ORB", "SIFT", "LBP", "Gabor", "Entropy", "LBP and Entropy"]
    res = ui.prompt("Choose a feature selection algorithm:", options)
    type = options[int(res)]

    data = []
    for img in pb.progressbar(images): # Process each image
        if type == "ORB":              # Corner features
            alg = cv2.ORB_create() 
            descriptor_size = 32
            vector_size = 32
            data.append(describe_keypoints(img, alg, vector_size, descriptor_size))
        else:
            print("ERROR: Type " + type + " not found (features.extract_features())\n")
            return 1

    return data
