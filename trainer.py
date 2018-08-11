#!/usr/bin/python3
import numpy as np
import preprocess as pproc
import features as ft
import random
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

def save_filenames(y_true, y_pred, filenames):
    """Save filenames sorted by confusion matrix"""
    tp = []
    tn = []
    fp = []
    fn = []
    prefix = str(random.randrange(1000000, 9999999))
    for i in range(len(y_true)):
        if not y_true[i] and not y_pred[i]:
            tp.append(filenames[i])
        elif y_true[i] and y_pred[i]:
            tn.append(filenames[i])
        elif not y_true[i] and y_pred[i]:
            fn.append(filenames[i])
        elif y_true[i] and not y_pred[i]:
            fp.append(filenames[i])
    np.savetxt("results/{}_true_pos.txt".format(prefix), tp, delimiter=",", fmt="%s")
    np.savetxt("results/{}_true_neg.txt".format(prefix), tn, delimiter=",", fmt="%s")
    np.savetxt("results/{}_false_pos.txt".format(prefix), fp, delimiter=",", fmt="%s")
    np.savetxt("results/{}_false_neg.txt".format(prefix), fn, delimiter=",", fmt="%s")
    return 0

def true_neg(y_true, y_pred, args=None):
    """Count number of true negatives"""
    fp_count = 0
    for i in range(len(y_true)):
        if not y_true[i] and not y_pred[i]:
            fp_count += 1
    return fp_count

def true_pos(y_true, y_pred, args=None):
    """Count number of true positives"""
    fp_count = 0
    for i in range(len(y_true)):
        if y_true[i] and y_pred[i]:
            fp_count += 1
    return fp_count

def false_pos(y_true, y_pred, args=None):
    """Count number of false positives"""
    fp_count = 0
    for i in range(len(y_true)):
        if not y_true[i] and y_pred[i]:
            fp_count += 1
    return fp_count

def false_neg(y_true, y_pred, args=None):
    """Count number of false negatives"""
    fn_count = 0
    for i in range(len(y_true)):
        if y_true[i] and not y_pred[i]:
            fn_count += 1
    return fn_count

def model_evaluation(data, targets, clf):    
    # Cross validate and calculate scores
    f_pos = skm.make_scorer(false_pos)
    f_neg = skm.make_scorer(false_neg)
    t_pos = skm.make_scorer(true_pos)
    t_neg = skm.make_scorer(true_neg)
    scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "f_pos": f_pos,
            "f_neg": f_neg,
            "t_pos": t_pos,
            "t_neg": t_neg,
    }
    scores = cross_validate(clf, data, targets, scoring=scoring, cv=10)
    print("Scores calculated from 10-fold cross validation:")
    print("Accuracy:  {},\t{}".format(round(np.mean(scores["test_accuracy"]),  4), scores["test_accuracy"]))
    print("Precision: {},\t{}".format(round(np.mean(scores["test_precision"]), 4), scores["test_precision"]))
    print("Recall:    {},\t{}".format(round(np.mean(scores["test_recall"]),    4), scores["test_recall"]))
    print("F1:        {},\t{}".format(round(np.mean(scores["test_f1"]),        4), scores["test_f1"]))
    print("False Pos: {},\t{}".format(round(np.mean(scores["test_f_pos"]),     4), scores["test_f_pos"]))
    print("False Neg: {},\t{}".format(round(np.mean(scores["test_f_neg"]),     4), scores["test_f_neg"]))
    print("True Pos:  {},\t{}".format(round(np.mean(scores["test_t_pos"]),     4), scores["test_t_pos"]))
    print("True Neg:  {},\t{}".format(round(np.mean(scores["test_t_neg"]),     4), scores["test_t_neg"]))

def train(data, targets, filenames):
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
        1: RandomForestClassifier(n_estimators=50, max_depth=None, random_state=0),
        2: DecisionTreeClassifier(random_state=0),
        3: KNeighborsClassifier()
    }
    clf = switch.get(int(res))

    if mode == "Cross validation":
        model_evaluation(data, targets, clf)
    elif mode == "Build and test model":
        # Train model
        clf.fit(data, targets)

        # Get test dir
        while True:
            dirname = ui.prompt("Which directory are the test files in?")
            if os.path.isdir(dirname):
                break
            print("ERROR: Directory not found.")

        # Set up data/targets for test model
        print("\n************************************")
        print("*  PREPARING MODEL FOR EVALUATION  *")
        print("************************************")

        pageNames, y_true, filenames = pproc.process(dirname)    
        y_true = [val == "INFEC" for val in y_true] # Set INFEC as positive val
        test_data = ft.features(pageNames)
   
        y_pred = clf.predict(test_data)

        save_filenames(y_true, y_pred, filenames)
    
        conf_matrix = skm.confusion_matrix(y_true, y_pred)
        accuracy = skm.accuracy_score(y_true, y_pred)
        precision = skm.precision_score(y_true, y_pred, average=None)
        recall = skm.recall_score(y_true, y_pred, average=None)
        f1 = skm.f1_score(y_true, y_pred, average=None)
        print("\n{}".format(conf_matrix))
        print("Accuracy:  {}".format(accuracy))
        print("Precision: {}".format(precision[1]))
        print("Recall:    {}".format(recall[1]))
        print("F1:        {}".format(f1[1]))
