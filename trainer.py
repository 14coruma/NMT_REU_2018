#!/usr/bin/python3
import numpy as np
import preprocess as pproc
import features as ft
import os
import ui

# ML Classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Model evaluation
from sklearn.model_selection import cross_validate
import sklearn.metrics as skm

def model_evaluation(data, targets, clf):
    # Cross validate and calculate scores
    scoring = ["accuracy", "precision", "recall", "f1"] # Choose scoring methods
    scores = cross_validate(clf, data, targets, scoring=scoring, cv=5)
    print("Scores calculated from 5-fold cross validation:")
    print("Accuracy:  {},\t{}".format(round(np.mean(scores["test_accuracy"]),  4), scores["test_accuracy"]))
    print("Precision: {},\t{}".format(round(np.mean(scores["test_precision"]), 4), scores["test_precision"]))
    print("Recall:    {},\t{}".format(round(np.mean(scores["test_recall"]),    4), scores["test_recall"]))
    print("F1:        {},\t{}".format(round(np.mean(scores["test_f1"]),        4), scores["test_f1"]))

def train(data, targets):
    targets = [val == "INFEC" for val in targets] # Set INFEC as positive val
   
    # Choose training mode
    options = ["Cross validation", "Build and test model"]
    res = ui.prompt(options=options)
    mode = options[int(res)]

    # Choose ML algorithm
    options = ["Support Vector Machine", "Random Forest",
            "Decision Tree Classifier", "KNN"]
    res = ui.prompt("Choose a ML algorithm:", options)
    switch = {
        0: svm.SVC(C=100., random_state=0),
        1: RandomForestClassifier(max_depth=3, random_state=0),
        2: DecisionTreeClassifier(random_state=0),
        3: KNeighborsClassifier()
    }
    clf = switch.get(int(res))

    if mode == "Cross validation":
        model_evaluation(data, targets, clf)
    elif mode == "Build and test model":
        clf.fit(data, targets)

        while True:
            res = ui.prompt("Which directory are the test files in?")
            if os.path.isdir(res):
                break
            print("ERROR: Directory not found.")

        pageNames, y_true = pproc.process(res)    
        y_true = [val == "INFEC" for val in y_true] # Set INFEC as positive val
        test_data = ft.features(pageNames)
    
        y_pred = clf.predict(test_data)
        
        f1 = skm.f1_score(y_true, y_pred, average=None)
        print("F1:        {},\t{}".format(round(np.mean(skm.f1_score(y_true, y_pred, average=None)), 4), f1))
