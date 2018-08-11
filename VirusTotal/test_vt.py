#!/usr/bin/python3
import sklearn.metrics as skm
import progressbar as pb
import numpy as np
import itertools
import requests
import hashlib
import sys
import os

import ui

MY_KEY = "d98dcf2ab3e64df99df3bb85ed2b98203795cabf83ab407450e40b4c0732c3ad"
MALWARE_THRESHOLD = 0.25
SCANNERS = ["Avast", "AVG", "McAfee", "Microsoft", "nProtect", "Symantec", "Webroot"]

def report(file_id):
    url = 'https://www.virustotal.com/vtapi/v2/file/report'
    params = {'apikey': MY_KEY, 'resource': file_id, 'allinfo': True}
    response = requests.get(url, params=params)
    return response.json()

def scan(filename, file):
    url = 'https://www.virustotal.com/vtapi/v2/file/scan'
    params = {'apikey': MY_KEY}
    files = {'file': (filename, file)}
    response = requests.post(url, files=files, params=params)
    return response.json()

def thresholdReport(res, item, targets, data, max_neg, min_pos):
    pos = res["positives"]
    percent_pos = pos / res["total"]
    pred = percent_pos < MALWARE_THRESHOLD
    data.append([pred, percent_pos, pos])
    if targets[-1] == "CLEAN" and data[-1][0]:
        if percent_pos >= max_neg[0]:
            max_neg = [percent_pos, res["total"], item]
    elif targets[-1] == "INFEC" and not data[-1][0]:
        if percent_pos <= min_pos[0]:
            min_pos = [percent_pos, res["total"], item]
    else:
        print("Incorrect: {}, target: {}, pred: {}".format(item, targets[-1], data[-1]))
        print("           {} / {} flagged".format(pos, res["total"]))
    return data, max_neg, min_pos


def main():
    options = ["Threshold", "Select scanners"]
    type = options[int(ui.prompt(options=options))]

    dir = sys.argv[1]
    queue = []
    data = []
    targets = []
    min_pos = [1, 0, None]
    max_neg = [0, 0, None]
    
    scans = {}
    for s in SCANNERS:
        scans[s] = []

    # Report / scan each file
    for item in pb.progressbar(os.listdir(dir)):
        if item.endswith(".bmp") or item.endswith(".png"):
            continue
        filename = os.path.join(dir, item)
        file = open(filename, 'rb').read()
        hash = hashlib.sha256(file).hexdigest()
        res = report(hash)
        if res["response_code"] == 0: # File not found
            scan(filename, file)
            queue.append([filename, hash])
        elif res["response_code"] == -2: # Scan in Queue
            queue.append([filename, hash])
        elif res["response_code"] == 1: # File found
            targets.append(item[:5])
            if type == "Threshold":
                data, max_neg, min_pos = thresholdReport(res, item, targets, data, max_neg, min_pos)
            elif type == "Select scanners":
                for s in SCANNERS:
                    if s in res["scans"]:
                        scans[s].append(not res["scans"][s]["detected"])
                    else:
                        scans[s].append(False)

    # Check up on items in the queue
    while len(queue) > 0:
        print("Left in Queue: {}".format(len(queue)))
        for item in queue:
            res = report(item[1])
            code = res["response_code"]
            if code == 0: # File not found
                print("ERROR: File {} not enqueued.".format(item[0]))
            elif code == 1: # File found
                targets.append(item[:5])
                if type == "Threshold":
                    data, max_neg, min_pos = thresholdReport(res, item, targets, data, max_neg, min_pos)
                elif type == "Select scanners":
                    for s in SCANNERS:
                        if s in res["scans"]:
                            scans[s].append(not res["scans"][s]["detected"])
                        else:
                            scans[s].append(False)
                queue.remove(item)
  
    if type == "Threshold":
        data = np.array(data)
        y_pred = data[:,0] # Just first column of data[]
        y_true = [val == "CLEAN" for val in targets] # Set INFEC as positive val

        conf_matrix = skm.confusion_matrix(y_true, y_pred)
        accuracy = skm.accuracy_score(y_true, y_pred)
        precision = skm.precision_score(y_true, y_pred, average=None)
        recall = skm.recall_score(y_true, y_pred, average=None)
        f1 = skm.f1_score(y_true, y_pred, average=None)
        print("\n{}".format(conf_matrix))
        print("Accuracy:  {}".format(accuracy))
        print("Precision: {}".format(precision[0]))
        print("Recall:    {}".format(recall[0]))
        print("F1:        {}".format(f1[0]))

        print("Max Negative: {}\nMin Positive: {}".format(max_neg, min_pos))
    else:
        for s in SCANNERS:
            print("\nResults for {}:".format(s));
            data = np.array(scans[s])
            y_pred = data
            y_true = [val == "CLEAN" for val in targets] # Set INFEC as positive val

            conf_matrix = skm.confusion_matrix(y_true, y_pred)
            accuracy = skm.accuracy_score(y_true, y_pred)
            precision = skm.precision_score(y_true, y_pred, average=None)
            recall = skm.recall_score(y_true, y_pred, average=None)
            f1 = skm.f1_score(y_true, y_pred, average=None)
            print(conf_matrix)
            print("Accuracy:  {}".format(accuracy))
            print("Precision: {}".format(precision[0]))
            print("Recall:    {}".format(recall[0]))
            print("F1:        {}".format(f1[0]))

if __name__ == "__main__":
    main()
