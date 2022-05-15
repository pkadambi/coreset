import numpy as np
from sklearn.neighbors import KernelDensity


'''
functions for evaluating the kde objects
'''

def fit_kde(xinput, bw, kernel='gaussian'):
    '''
    xinput - the input data to run the kde on. [nsamples x nfeatures]
    bw - kde bandwidth
    '''

    return KernelDensity(kernel=kernel, bandwidth=bw).fit(xinput.reshape(-1, 1))


def evaluate_kde_proba(xdata, kdeobject):
    '''
    returns the probabilites from the kde object on xdata
    '''
    if len(xdata.shape) < 2:
        _xdata = xdata.reshape(-1, 1)
    else:
        _xdata = xdata

    probas = np.exp(kdeobject.score_samples(_xdata))
    return probas


def compute_kde_c1_posterior(c0kde_object, c1kde_object, xinput):
    '''
    Given the c0kde and the c1kde this function predicts the probability on unseen data

    c0kde object - computes p(y=0|x), must be fit already trained
    c1kde object - computes p(y=1|x), must be fit already trained

    xinput - [nsamples x nfeatures]

    output of this function
    - a numpy array [nsamples x 1]
    '''
    if len(xinput.shape) < 2:
        _xdata = xinput.reshape(-1, 1)
    else:
        _xdata = xinput
    p_y0 = evaluate_kde_proba(_xdata, c0kde_object)
    p_y1 = evaluate_kde_proba(_xdata, c1kde_object)
    p_y1_x = p_y1 / (p_y0 + p_y1)
    return p_y1_x


'''
Functions for the data generating processes
'''


def posterior_pdf2(xinput):
    '''
    xinput - [nsamples x nfeatures], the x value for the data
    youtput - returns the probability that y=1
    '''
    assert not any(xinput > 1) and not any(
        xinput < 0), "This posterior function expects x values ONLY in the range 0<x<1"

    youtput = np.zeros((xinput.shape[0], 1))
    range1 = xinput < .2
    range2 = np.logical_and(xinput >= .20, xinput <= .6)
    range3 = xinput > .6

    youtput[range1] = .125 * np.sin(200 * xinput[range1] + 1) + .75
    youtput[range2] = .0625 * np.sin(20 * xinput[range2]) + .75
    youtput[range3] = .5
    return youtput


def assign_labels(xinput, yposterior):
    ylabels = np.zeros_like(yposterior)
    for ii, (xsamp, posteriorval) in enumerate(zip(xinput, yposterior)):
        ylabels[ii] = float(np.random.rand() < posteriorval)
    return ylabels