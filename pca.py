""" PCA transformation """

import numpy as np
from numpy import linalg as la

def prepare(data, n_components):
    ''' Prepares transformation '''
    x = np.matrix(data)
    mu = x.mean(0)
    cmx = np.cov(x, rowvar=False)
    evl, evc = la.eig(cmx)
    eid = np.fliplr([np.argsort(evl)])[0]

    for i in range(0, len(evc)):
        evc[i] = evc[i][eid]

    trmx = np.delete(evc, range(n_components, len(evl)), 1)
    return (mu, trmx)

def transform(data, mu, trmx, kernel):
    ''' Transforms data '''
    n = len(data)
    c = len(trmx[0])

    data_transformed = kernel(data, np.transpose(trmx))

    return data_transformed
