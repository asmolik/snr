import time
import os
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import const
import log


#http://stackoverflow.com/questions/24595153/is-it-possible-to-tune-parameters-with-grid-search-for-custom-kernels-in-scikit


# Wrapper class for the custom kernel chi2_kernel
class Chi2Kernel(BaseEstimator,TransformerMixin):
    def __init__(self, gamma=1.0):
        super(Chi2Kernel,self).__init__()
        self.gamma = gamma

    def transform(self, X):
        return chi2_kernel(X, self.X_train_, gamma=self.gamma)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


def parameters_search(score, params, data, pair, n_jobs=8):

    # Create a pipeline where our custom predefined kernel Chi2Kernel
    # is run before SVC.
    pipe = Pipeline([
        ('chi2', Chi2Kernel()),
        ('svm', svm.SVC()),
    ])

    A_train, y_train, A_test, y_test = data

    log.line()
    print("# Tuning hyper-parameters for %s\n" % score)
    clf = GridSearchCV(pipe,
                       params,
                       cv=5,
                       scoring=score,
                       n_jobs=n_jobs)
    start = time.time()
    clf.fit(A_train, y_train)
    log.time(time.time() - start)

    print("Best parameters set found on development set:\n%s\n" % clf.best_params_)
    print("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    y_pred = clf.predict(A_test)
    report = classification_report(y_test, y_pred)
    print(report)

    with open(const.RESULTS_FILE_NAME, "a") as file:
        file.write(str(pair) + "\n")
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            file.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
        file.write("\n%s\n\n" % clf.best_params_)
        file.write(report + "\n\n")

        return clf
