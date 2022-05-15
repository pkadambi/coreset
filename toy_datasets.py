import numpy as np
import matplotlib.pyplot as plt

def gen_simple_binary_data(mu_x = None, mu_y = None, N=1000, ndims=2):
    
    x0 = np.random.randn(N, ndims) 
    
    if mu_x is not None: #if mu_x not supplied, it's centered at zero
        x0 = x0 + mu_x
        
    x1 = np.random.randn(N, ndims)
    if mu_y is None:
        x1 = x1 + np.ones([N, ndims])
    else:
        x1 = np.random.randn(N, ndims) + mu_y
        
    labels = np.concatenate([np.zeros(N), np.ones(N)])
    
    return x0, x1, labels
    
def gen_linear_dataset(xmin=-1, xmax=1, ymin=-1, ymax=1, ndim=2, N=1000, sigma=.1):
    xdata = np.random.rand(N, ndim)
    xdata[:,0] = xdata[:,0] * (xmax - xmin) - (xmax+xmin)/2
    xdata[:,1] = xdata[:,1] * (ymax - ymin) - (ymax+ymin)/2
    labels = xdata[:,0]<(xdata[:,1] + sigma * np.random.randn(N))
    return xdata, labels
    

def inscribed_circle_dataset(N=1000, radius=1, sidelen=2, sigma=.1):
    xdata = sidelen * (np.random.rand(N, 2) - .5)
    circleinds = xdata[:,0] ** 2 + xdata[:,1] ** 2 <= (radius **2 + sigma * np.random.randn(len(xdata)))
    c0data = xdata[~circleinds]
    c1data = xdata[circleinds]
    labels = np.concatenate([np.zeros(len(c0data)), np.ones(len(c1data))])
    return c0data, c1data, labels


def plot_2d_dataset(xdata, labels, title):
    plt.figure()
    plt.scatter(xdata[labels==1, 0], xdata[labels==1, 1], color='r', alpha=.25,  marker='8', label='Class 1')
    plt.scatter(xdata[labels==0, 0], xdata[labels==0, 1], color='b', alpha=.25, marker='8', label='Class 0')
    plt.title(title)
    plt.grid()
    plt.legend()
    
    