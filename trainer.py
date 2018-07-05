#!/usr/bin/python3
import numpy as np

# ML Classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Model evaluation
from sklearn.model_selection import cross_validate

import ui

def train(data, targets):
    options = ["Support Vector Machine", "Random Forest",
            "Decision Tree Classifier", "KNN"]
    res = ui.prompt("Choose a ML algorithm:", options)
    switch = {
        0: svm.SVC(C=100.),
        1: RandomForestClassifier(max_depth=2),
        2: DecisionTreeClassifier(),
        3: KNeighborsClassifier()
    }
    clf = switch.get(int(res))

    # Cross validate and calculate scores
    scoring = ["accuracy", "precision", "recall", "f1"] # Choose scoring methods
    targets = [val == "INFEC" for val in targets] # Set INFEC as positive val
    scores = cross_validate(clf, data, targets, scoring=scoring, cv=5)
    print("Scores calculated from 5-fold cross validation:")
    print("Accuracy:  {},\t{}".format(round(np.mean(scores["test_accuracy"]),  4), scores["test_accuracy"]))
    print("Precision: {},\t{}".format(round(np.mean(scores["test_precision"]), 4), scores["test_precision"]))
    print("Recall:    {},\t{}".format(round(np.mean(scores["test_recall"]),    4), scores["test_recall"]))
    print("F1:        {},\t{}".format(round(np.mean(scores["test_f1"]),        4), scores["test_f1"]))