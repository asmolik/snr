
import time
from functools import partial
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.metrics.pairwise as metrics
import log
import const
import pca


class PcaPoly(BaseEstimator,TransformerMixin):
    def __init__(self, n_components, degree=2.0, coef0=1.0):
        super(PcaPoly,self).__init__()
        self.n_components = n_components
        self.kernel = partial(metrics.polynomial_kernel, degree=degree, coef0=coef0)

    def transform(self, X):
        m, t = pca.prepare(X, self.n_components)
        return pca.transform(X, m, t, self.kernel)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


class PcaRbf(BaseEstimator,TransformerMixin):
    def __init__(self, n_components, gamma=1.0):
        super(PcaRbf,self).__init__()
        self.n_components = n_components
        self.kernel = partial(metrics.rbf_kernel, gamma=gamma)

    def transform(self, X):
        m, t = pca.prepare(X, self.n_components)
        return pca.transform(X, m, t, self.kernel)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


def poly_param_search_pipe(n_components, C=1, coef0=1, degree=2):
    return Pipeline([
        ('pca_poly', KernelPCA(n_components, kernel="poly", coef0=coef0, degree=degree)),
        ('svm', svm.SVC(C=C))
    ])


def rbf_param_search_pipe(n_components, C=1, gamma=1):
    return Pipeline([
        ('pca_rbf', KernelPCA(n_components, kernel="rbf", gamma=gamma)),
        ('svm', svm.SVC(C=C))
    ])


def poly_pipe(n_components, C=1, coef0=1, degree=2):
    return Pipeline([
        ('pca_poly', PcaPoly(n_components, coef0=coef0, degree=degree)),
        ('svm', svm.SVC(C=C))
    ])


def rbf_pipe(n_components, C=1, gamma=1):
    return Pipeline([
        ('pca_rbf', PcaRbf(n_components, gamma=gamma)),
        ('svm', svm.SVC(C=C))
    ])


def parameters_search(score, params, pipe, data, pair, n_jobs=8):

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

    with open(const.RESULTS_FILE_NAME_KERNEL_PCA, "a") as file:
        file.write(str(pair) + "\n")
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            file.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
        file.write("\n%s\n\n" % clf.best_params_)
        file.write(report + "\n\n")

    return clf