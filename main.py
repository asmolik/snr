""" Main module """

import time
from functools import partial
import read_data as d
import transform
import const
import log
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def load_data(feature, classes, components_cnt):
    log.line()
    log.dataset_configuration(feature, classes, components_cnt)
    log.message('Loading data...')

    start = time.time()
    train_data, train_labels, test_data, test_labels = d.read_test_and_train_data(feature, classes)
    train_data, test_data = transform.perform_pca(train_data, test_data, components_cnt)
    train_data, test_data = transform.normalize_data(train_data, test_data)
    log.time(time.time() - start)

    log.dataset_summary(len(train_data), len(test_data))
    return train_data, train_labels, test_data, test_labels

def evaluate_kernel(svc, name, data):
    train_data, train_labels, test_data, test_labels = data
    log.message('Training {} kernel ...'.format(name))

    start = time.time()
    svc.fit(train_data, train_labels)
    log.time(time.time() - start)

    log.message('Evaluating ...')
    predicted_labels = svc.predict(test_data)
    log.message(classification_report(test_labels, predicted_labels))

def evaluate_chi2_kernel(data, gamma, C):
    train_data, train_labels, test_data, test_labels = data
    log.message('Training chi^2 kernel ...')

    start = time.time()
    svc = svm.SVC(kernel='precomputed', C=C)
    train_kernel = chi2_kernel(train_data, gamma=gamma)
    svc.fit(train_kernel, train_labels)
    log.time(time.time() - start)

    log.message('Evaluating ...')
    test_kernel = chi2_kernel(test_data, train_data, gamma=gamma)
    predicted_labels = svc.predict(test_kernel)
    log.message(classification_report(test_labels, predicted_labels))

# http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py

def parameters_search(score, params, data, n_jobs=8):

    train_data, train_labels, test_data, test_labels = data

    log.line()
    print("# Tuning hyper-parameters for %s\n" % score)
    clf = GridSearchCV(
        svm.SVC(C=1),
        params,
        cv=5,
        scoring=score,
        n_jobs=n_jobs
    )
    start = time.time()
    clf.fit(train_data, train_labels)
    log.time(time.time() - start)

    print("Best parameters set found on development set:\n%s\n" % clf.best_params_)
    print("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print(
        "\nDetailed classification report:\n"
        "The model is trained on the full development set.\n"
        "The scores are computed on the full evaluation set.\n"
    )
    pred_labels = clf.predict(test_data)
    print(classification_report(test_labels, pred_labels))

    return clf

####################################### MAIN SCRIPT #######################################

# np.random.permut

CLASSES = list(range(1, 3)) # Use (0, 43) for full dataset
COMPONENTS_CNT = 256
FEATURE = const.HOG01

if __name__ == '__main__':

    data = load_data(FEATURE, CLASSES, COMPONENTS_CNT)

    kernels = [
        (svm.SVC(kernel='linear', C=0.1), 'linear'),
        (svm.SVC(kernel='poly', degree=2, C=0.1, gamma=0.1, coef0=10.0), 'poly2'),
        (svm.SVC(kernel='poly', degree=3, C=0.1, gamma=0.1, coef0=0.0), 'poly3'),
        (svm.SVC(kernel='rbf', C=10.0, gamma=0.1), 'rbf')
    ]

    # Takes roughly 1-2 min per kernel for full dataset
    # Best result for current parameters gives rbf - 94%, others are ~ 93%
    for k in kernels:
        evaluate_kernel(k[0], k[1], data)

    # # Takes ~35 min for full dataset (very memory expensive)
    # # Results are on 93-94% level
    for gamma, C in [(0.1, 10.0)]:
        evaluate_chi2_kernel(data, gamma, C)

    kernel_params = {
        'linear': {
            'kernel': ['linear'],
            'C': [0.1, 1, 10]
        },
        'poly2': {
            'kernel': ['poly'],
            'degree': [2],
            'coef0': [0, 0.1, 1, 10],
            'gamma': [0.1, 1, 10],
            'C': [0.1, 1, 10]
        },
        'poly3': {
            'kernel': ['poly'],
            'degree': [3],
            'coef0': [0, 0.1, 1, 10],
            'gamma': [0.1, 1, 10],
            'C': [0.1, 1, 10]
        },
        'rbf': {
            'kernel': ['rbf'],
            'gamma': [0.1, 1, 10],
            'C': [0.1, 1, 10]
        }
    }

    SCORE = 'accuracy' # Najlepiej wybrac jeden w miare ogolny score, zeby zaoszczedzic obliczen
    # Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
    # 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
    # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',
    # 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
    # 'precision_weighted', 'r2', 'recall', 'recall_macro',
    # 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

    N_JOBS = const.N_THREADS # Set to number of processor threads for the best performance

    # Takes ~2 min for 2 classes
    parameters_search(SCORE, kernel_params['rbf'], data, N_JOBS)
