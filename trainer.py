#!/usr/bin/python3

# ML Classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
