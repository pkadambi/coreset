import numpy as np
from sklearn.neighbors import NearestNeighbors

''' Generate Posterior Distribution '''

def oracle_sinusoid_posterior(query):
    if np.abs(query[0])>4 or np.abs(query[1])>4:
        return Exception('Absolute value of each coordinate must be less than 4')
    
    if query[0]<-1:
        xpeicewise = (query[0] + 4)/2
        posterior = np.cos(2 * np.pi * 2 * xpeicewise)
        posterior = .4 * posterior + .5
    
    elif query[0]<=2:
        xpeicewise = (query[0] + 2)/2
        posterior = np.cos(2 * np.pi * 2 * xpeicewise)
        posterior = .2 * posterior + .5           
    
    elif query[0]<=4:
        posterior = -(query[0] - 4) * (.6-.4)/2 + .4
        
    return posterior

#TODO: put these into some kind of posterior estimator class, make all of these static methods
def estimate_posterior(data, labels):
    '''
    data - [n_points, n_dimensions]
    labels - [n_points, 2]
    '''
    
    n_points = data.shape[0]

    return None

def simple_knn_posterior_estimate(data, labels, k, kernel=None):
    '''
    data - [n_points, n_dimensions]
    labels - [n_points, 2]
    '''
    
    def _compute_ber(neighbdata, neighborlabels, centerlabel):
        errors = np.sum(np.abs(neighborlabels - centerlabel))
#         print(errors)
        N = np.sum(neighborlabels==0)
        M = np.sum(neighborlabels==1)
        if N==0 or M==0:
            kerneldp = 1.
            kernelBayesLB = 0.
        else:
            kerneldp = 1 - ((M + N) / (2 * M * N)) * errors
            if kerneldp<=0:
                kerneldp=0.
                kernelBayesLB = 0
            else:
                kernelBayesLB = .5 - .5 * np.sqrt(kerneldp)
        return kernelBayesLB
    
    def _compute_1nn_threshold():
        nbrs = NearestNeighbors(n_neighbors = k + 1).fit(data)
        distances, inds = nbrs.kneighbors(data)
        
        
    n_points = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors = k + 1).fit(data)
    distances, inds = nbrs.kneighbors(data)
    posterior_value = np.zeros([n_points,1])
    ber_estimates = np.zeros([n_points, 1])
    posterior_var = np.zeros([n_points,1])
    posterior_thresh = np.zeros([n_points, 1])
    ber_var = np.zeros([n_points, 1])
    
    for ii in range(n_points):
        neighblabels = labels[inds[ii, ]]
        neighbdata = data[inds[ii,:]]
        posterior_value[ii] = sum(neighblabels[1:])/k
        ber_estimates[ii] = _compute_ber(neighbdata[1:], neighblabels[1:], neighblabels[0])
    
    for ii in range(n_points):
        neighb_ber = ber_estimates[inds[ii, :]]
        neighb_posterior= posterior_value[inds[ii, :]]
        ber_var[ii] = np.var(neighb_ber[1:])
        posterior_var[ii] = np.var(neighb_posterior[1:])
        posterior_thresh[ii] = sum(neighb_posterior[1:]-neighb_posterior[0]>(1/k))
    return posterior_value, ber_estimates, posterior_var, ber_var, posterior_thresh
