import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from numpy.linalg import norm
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
def calc_crossentropy(p, q):
    '''
    p - [n_samples x n_classes]
    q - [n_samples x n_classes]
    '''
    h_p = entropy(p.T)
    dkl = entropy(p.T, q.T)
    return h_p + dkl

def calc_kneighborhood_xent(yi, yjs):
    '''
    yi - [1 x n_classes], the center point
    yjs - [n_k x n_classes], the k neighborhood
    '''
    n_k = yjs.shape[0]
    yi_repeated = np.repeat(yi, n_k)
    return calc_crossentropy(yi_repeated, yjs)

def calc_neighborhood_lipschitz_constant(xi, xjs, yi, yjs):
    '''
    xi - [1 x n_features]
    xjs - [n_k x n_features] (where n_k is the size of the knn neighborhood)
    yi - [1 x 2]
    yjs - [n_k x n_features]
    '''
    n_k = yjs.shape[0]
    xi_rep = np.repeat(xi, n_k, axis=1)
    yi_rep = np.repeat(yi, n_k, axis=1)
    x_l1_norms = norm(xi - yi_rep, ord=1)
    y_l1_norms = norm(yi_rep - yjs, ord=1, axis=0)
    classwise_lipschitz_constants = y_l1_norms/x_l1_norms    
    
    return classwise_lipschitz_constants
    

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

def xentropy_knn_posterior_estimate(data, softmax_labels, k, kernel=None):
    '''
    data - [n_points, n_dimensions]
    labels - [n_points, 2]
    '''
    import torch
    xent = torch.nn.CrossEntropyLoss()
    n_points = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors = k + 1).fit(data)
    distances, inds = nbrs.kneighbors(data)
    xentropies = np.zeros([n_points,1])
    
    for ii in range(n_points):
        neib_inds = inds[ii, 1:]
        neighb_labels = softmax_labels[neib_inds]
        xi_label = softmax_labels[ii, :]        
        xentropies[ii] = calc_kneighborhood_xent(xi_label, neighb_labels)
#         posterior_var[ii] = np.var(neighb_posterior[1:])
        
    return xentropies


"""
Set up unlabled data 
"""
import matplotlib

# fraction_query = .075
# MC_ITERS = 10
# n_seed = 250
# n_query = 100
# nplots = 0
# PLOT = True
# PLOT = False
# BUDGET = 5000

# def training_run(xtrain, ytr_label, sample_weights, tr_sfmx, PLOT = False, BUDGET = 5000,
#                  MC_ITERS = 10, n_seed = 250, n_query = 100, nplots = 0):
#     import tqdm
#     active_accs = []
#     active_epes = []
#     active_xents = []
#     for MC in tqdm.tqdm(range(MC_ITERS)):
#         xseed, seed_inds = uniform_sample(xdata=xtrain, ylogits=ytr_label, nsample=n_seed)
#         selected_inds = seed_inds
#         # unlabled_inds = np.delete(np.arange(len(xtrain)), selected_inds)
#
#
#         active_sampsize = []
#         active_acc = []
#         active_epe = []
#         active_label_xent = []
#         active_teacher_xent = []
#
#         while len(selected_inds) < BUDGET:
#             unlabled_inds = np.delete(np.arange(len(xtrain)), selected_inds)
#             xunlabel = xtrain[unlabled_inds]
#             ytr_unlabled = ytr_label[unlabled_inds]
#             tr_unlabled_posterior = tr_sfmx[unlabled_inds]
#
#             xselected = xtrain[selected_inds]
#             yselected = ytr_label[selected_inds]
#             posterior_selected = tr_sfmx[selected_inds]
#
#             xstage1 = xselected
#             ystage1 = yselected
#             posterior_stage1 = posterior_selected
#
#             # xtrain, tr_posterior, ytr_label = generate_samples(n_samples=N)
#             mu = np.mean(xstage1, axis=0)
#             std = np.std(xstage1, axis=0)
#         #     mu_std.append([mu, std])
#             _xstage1 = (xstage1 - mu)/std
#             classifier = train_classifier(xtrain=_xstage1, ytrain=ystage1, classifier='mlp')
#             # _xstage1
#             _tr_acc = evaluate_classifier(classifier, _xstage1, ystage1, accuracy)
#             _tr_mse = evaluate_classifier(classifier, _xstage1, ystage1, mse)
#
#             _xtest = (xtest - mu)/std
#             test_epe =  evaluate_classifier(classifier, _xtest, yte_label, accuracy)
#             test_acc =  evaluate_classifier(classifier, _xtest, yte_label, mse)
#             test_xent = evaluate_classifier(classifier, _xtest, te_posterior, xentropy)
#
#             active_sampsize.append(len(selected_inds))
#             active_acc.append(1-test_acc)
#             active_epe.append(test_epe)
#             active_label_xent.append(test_xent)
#             active_teacher_xent.append()
#             print('active training, started w/ %d data samples, queried: %d, total %d ' % (len(selected_inds)-n_query,
#                                                                                            n_query, len(selected_inds)))
#             print('Active dataset EPE', test_epe)
#             print('active dataset acc', 1-test_acc)
#             print('active dataset xent', test_xent)
#
#
#
#             """
#             Select Samples (used in the next iteration)
#             """
#
#
#             pred_logits = classifier.predict_proba(_xtrain)
#     #         proba_noise = np.random.randn()
#             min_logit_value_inds = argmin_logit_uncertainty_rank(xdata=xtrain, predicted_probas=pred_logits, randomize=False,
#                                                                        top_k=n_query, ignore_inds=selected_inds, uniform_frac=.15)
#
#
#             if nplots<15 and PLOT:
#                 plt.figure(figsize=(18, 4))
#                 plt.subplot(1, 3, 1)
#                 plt.scatter(xtrain[selected_inds, 0], xtrain[selected_inds, 1], marker='.',
#                             c=classifier.predict_proba(_xtrain)[selected_inds, 1], vmin=0., vmax=1.)
#     #                         c=classifier.predict_proba(_xtrain)[selected_inds, 1], cmap='hsv', vmin=0., vmax=1.)
#     #             plt.scatter(xtrain[min_logit_value_inds, 0], xtrain[min_logit_value_inds, 1],
#     #                         c=np.ones_like(min_logit_value_inds), cmap='Greys', vmin=0., vmax=1., label='Selected')
#                 plt.grid()
#     #             plt.colorbar()
#     #             plt.colorbar(matplotlib.cm.ScalarMappable(cmap='hsv'))
#                 plt.colorbar()
#
#                 plt.title('Training Data Predicted Probas, %d Samp Size' % len(selected_inds))
#                 plt.ylim(-4.25, 4.25)
#                 plt.xlim(-4.25, 4.24)
#     #             plt.ylim(-2, 0)
#     #             plt.xlim(-2, 0)
#     #             plt.figure()
#                 plt.subplot(1, 3, 2)
#     #             plt.scatter(xtrain[selected_inds, 0], xtrain[selected_inds, 1],
#     #                         c=classifier.predict_proba(_xtrain)[selected_inds, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
#     #             plt.scatter(xtrain[:, 0], xtrain[:, 1], marker='.',
#     #                         c=classifier.predict_proba(_xtrain)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
#     #                         c=classifier.predict_proba(_xtrain)[:, 1], vmin=0., vmax=1., label='Train')
#     #             wts = classifier.predict_proba(_xtrain)[:, 1]
#     #             inds = np.argwhere(np.abs(wts-.5)<.1)
#     #             print(len(inds))
#                 plt.scatter(xtest[:, 0], xtest[:, 1],
#                             c=classifier.predict_proba(_xtest)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Test')
#     #                         c=classifier.predict_proba(_xtest)[:, 1], vmin=0., vmax=1., label='Test')
#
#     #             plt.scatter(xtrain[min_logit_value_inds, 0], xtrain[min_logit_value_inds, 1],
#     #                         c=np.ones_like(min_logit_value_inds), cmap='Greys', vmin=0., vmax=1., label='Selected')
#                 plt.legend()
#                 plt.grid()
#     #             plt.colorbar(matplotlib.cm.ScalarMappable(cmap='hsv'))
#                 plt.title('All Available Points (From Prev Round)')
#                 plt.colorbar()
#                 plt.ylim(-4.25, 4.25)
#                 plt.xlim(-4.25, 4.24)
#     #             plt.ylim(-2, 0)
#     #             plt.xlim(-2, 0)
#                 plt.subplot(1, 3, 3)
#     #             plt.scatter(xtrain[selected_inds, 0], xtrain[selected_inds, 1],
#     #                         c=classifier.predict_proba(_xtrain)[selected_inds, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
#     #             plt.scatter(xtrain[:, 0], xtrain[:, 1],
#     #                         c=classifier.predict_proba(_xtrain)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
#     #             plt.scatter(xtest[:, 0], xtest[:, 1],
#     #                         c=classifier.predict_proba(_xtest)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Test')
#
#     #             pred_logits = classifier.predict_proba(_xtrain)
#     #             probainds = np.argmax(pred_logits, axis=1)
#     #             largest_logit = pred_logits[np.arange(len(pred_logits)), probainds]
#     #             sortinds = np.argsort(largest_logit)
#     #             print(len(sortinds))
#     #             sortinds = np.delete(sortinds, selected_inds)
#     #             print(len(sortinds))
#     #             if ignore_inds is not None:
#     #                 sortinds = np.delete(sortinds, ignore_inds)
#     #             inds = sortinds[:n_query]
#
#                 plt.scatter(xtrain[min_logit_value_inds, 0], xtrain[min_logit_value_inds, 1], marker='.',
#                             c=np.ones_like(min_logit_value_inds), cmap='Greys', vmin=0., vmax=1., label='Selected')
#                 plt.legend()
#                 plt.grid()
#     #             plt.colorbar(matplotlib.cm.ScalarMappable(cmap='hsv'))
#                 plt.title('Selected Points')
#                 plt.colorbar()
#                 plt.ylim(-4.25, 4.25)
#                 plt.xlim(-4.25, 4.24)
#     #             plt.ylim(-2, 0)
#     #             plt.xlim(-2, 0)
#                 plt.show()
#                 nplots+=1
#     #         if nplots>10:
#     #             print('asdf')
#     #             print(jkj094)
#
#             selected_inds = np.concatenate([selected_inds, min_logit_value_inds])
#
#         active_epes.append(active_epe)
#         active_accs.append(active_acc)
#         active_xents.append(active_xent)
#     return active_accs, active_epes, active_xents

