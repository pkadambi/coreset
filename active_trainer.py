import numpy as np
import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from numpy.linalg import norm
import matplotlib.pyplot as plt
# from 
from densityestimate import *

from classifier_utils import *
def calc_crossentropy(p, q):
    '''
    p - [n_samples x n_classes]
    q - [n_samples x n_classes]
    '''
    h_p = entropy(p.T)
    dkl = entropy(p.T, q.T)
    return h_p + dkl

def calc_kldiv(p, q):
    '''
    p - [n_samples x n_classes]
    q - [n_samples x n_classes]
    '''
    return entropy(p.T, q.T)

def calc_jsd(p, q):
    '''
    p - [n_samples x n_classes]
    q - [n_samples x n_classes]
    '''
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)    

def calc_neighborhood_xent(yi, yjs):
    '''
    yi - [1 x n_classes], the center point
    yjs - [n_k x n_classes], the k neighborhood
    '''
    n_k = yjs.shape[0]
    yi_repeated = np.repeat(yi, n_k, axis=0)
#     print(yi.shape)
#     print(yi_repeated.shape)
#     print(yjs.shape)
    return np.mean(calc_crossentropy(yi_repeated, yjs))


def calc_neighborhood_kldiv(yi, yjs):
    '''
    yi - [1 x n_classes], the center point
    yjs - [n_k x n_classes], the k neighborhood
    '''
    n_k = yjs.shape[0]
    yi_repeated = np.repeat(yi, n_k, axis=0)
#     print(yi.shape)
#     print(yi_repeated.shape)
#     print(yjs.shape)
    return np.mean(calc_kldiv(yi_repeated, yjs))

def calc_neighborhood_jsd(yi, yjs):
    '''
    yi - [1 x n_classes], the center point
    yjs - [n_k x n_classes], the k neighborhood
    '''
    n_k = yjs.shape[0]
    yi_repeated = np.repeat(yi, n_k, axis=0)
#     print(yi.shape)
#     print(yi_repeated.shape)
#     print(yjs.shape)
    return np.mean(calc_jsd(yi_repeated, yjs))

def calc_neighborhood_lipschitz_constant(xi, xjs, yi, yjs):
    #ONLY WORKS FOR 2 CLASS 
    '''
    xi - [1 x n_features]
    xjs - [n_k x n_features] (where n_k is the size of the knn neighborhood)
    yi - [1 x 2]
    yjs - [n_k x n_features]
    '''
    n_k = yjs.shape[0]
    xi_rep = np.repeat(xi, n_k, axis=0)
    yi_rep = np.repeat(yi, n_k, axis=0)
#     print(xi_rep.shape)
#     print(xjs.shape)
#     print(yi_rep.shape)
#     print(yjs.shape)
    x_l1_norms = norm(xi_rep - xjs, ord=1, axis=0)
    y_l1_norms = norm(yi_rep - yjs, ord=1, axis=0)
#     print(x_l1_norms.shape)
#     print(y_l1_norms.shape)
    classwise_lipschitz_constants = y_l1_norms/x_l1_norms    
    
    return sum(classwise_lipschitz_constants)

def compute_neighbors(data, k):
    n_points = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors = k + 1).fit(data)
    distances, inds = nbrs.kneighbors(data)
    return distances, inds

def get_sampling_distributions(data, softmax_labels, k, knn_inds=None):
    #TODO: Make this work for multiclass (it already does for crossentropy, but Lipschitz does not)
    '''
    data - [n_points, n_dimensions]
    labels - [n_points, n_classes]
    k - number of nearest neighbors
    knn_inds - square symmetric matrix containing indicies of k nearest neighbors
    
    returns a set of sampling distributions 
    '''
    n_points = data.shape[0]
    xentropies = np.zeros([n_points,1])
    kldivs = np.zeros([n_points, 1])
    variances = np.zeros([n_points, 1])
    entropies = np.zeros([n_points, 1])
    lipschitzes = np.zeros([n_points, 1]) 
    jsd = np.zeros([n_points, 1])
#     bers = np.zeros()
    #TODO: Should BER be part of this list?
    
    if knn_inds is None:
        distances, knn_inds = compute_neighbors(data, k)

    for ii in range(n_points):
        neib_inds = knn_inds[ii, 1:]
        
        _xi = data[ii, :].reshape(1, -1)
        _xjs = data[neib_inds, :]
        yjs_neighb_labels = softmax_labels[neib_inds]
        yi_label = softmax_labels[ii, :].reshape(1, -1)
        
        xentropies[ii] = calc_neighborhood_xent(yi_label, yjs_neighb_labels)
        jsd[ii] = calc_neighborhood_jsd(yi_label, yjs_neighb_labels)
        
        kldivs[ii] = calc_neighborhood_kldiv(yi_label, yjs_neighb_labels)
        variances[ii] = np.var(yjs_neighb_labels[:,0])
        entropies[ii] = entropy(yi_label.T)
        lipschitzes[ii] = calc_neighborhood_lipschitz_constant(_xi, _xjs,
                                                               yi_label, yjs_neighb_labels)
#         break
    sampling_distributions = {'variance': variances, 'xent': xentropies, 'jsd': jsd, 
                              'kldiv':kldivs, 'entropy': entropies, 'lipschitz': lipschitzes}
    return sampling_distributions

''' Step 2: Compare with uncertainty sampling'''
from scipy.stats import entropy

def argmin_logit_uncertainty_rank(xdata, predicted_probas, top_k, randomize=False, ignore_inds=None, uniform_frac=None):
    probainds = np.argmax(predicted_probas, axis=1)
    largest_logit = predicted_probas[np.arange(len(predicted_probas)), probainds]
    
    if randomize:
        largest_logit = largest_logit + np.random.randn(len(largest_logit))/120
    sortinds = np.argsort(largest_logit)
    
    if ignore_inds is not None:
        sortinds = np.delete(sortinds, ignore_inds)
        
    n_unif = 0 if uniform_frac is None else int(uniform_frac * top_k)     
    n_select = top_k - n_unif    
    
    selected_inds = sortinds[:n_select]
    if n_unif>0:
#         sortinds = np.delete(sortinds, selected_inds)
        # from the array of indices, select elements uniformly at random
        unif_inds = np.random.choice(sortinds[n_select:], n_unif, replace=False)
        selected_inds = np.concatenate([selected_inds, unif_inds])
    return selected_inds

    
#TODO: rewrite the function below with better variable names and a more compressed format
def weight_based_sample(sampling_distribution, top_k, randomize=False, ignore_inds=None, uniform_frac=None):
    n_samples = len(sampling_distribution)
    all_inds = np.arange(n_samples)
    relevant_sampling_distribution = sampling_distribution
    relevant_sampling_distribution[ignore_inds] = 0.
    relevant_sampling_distribution = relevant_sampling_distribution/sum(relevant_sampling_distribution)
    
    n_unif = 0 if uniform_frac is None else int(uniform_frac * top_k)     
    n_select = top_k - n_unif    
    
    selected_inds = np.random.choice(all_inds, n_select, replace=False, p=relevant_sampling_distribution)
    
    if n_unif>0:
        unif_sampling_distribution = np.ones(n_samples)/n_samples
        unif_sampling_distribution[selected_inds] = 0
        unif_sampling_distribution[ignore_inds] = 0
        unif_sampling_distribution = unif_sampling_distribution/sum(unif_sampling_distribution)
        unif_inds = np.random.choice(all_inds, n_unif, replace=False, p=unif_sampling_distribution)
#         unif_inds = np.random.choice(sortinds[n_select:], n_unif, replace=False, p=sample_weights)
        selected_inds = np.concatenate([selected_inds, unif_inds])
        
    return selected_inds


def entropy_rank(xdata, ylogits, top_k):
    entropies = entropy(ylogits, axis=1)
    sortinds = np.argsort(entropies)
    selected_inds = sortinds[:top_k]
    selected_data = xdata[selected_inds]
    return selected_inds, selected_data

def uniform_sample(xdata, nsample):
    sortinds = np.random.choice(np.arange(len(xdata)), nsample, replace=False)
    return xdata[sortinds], sortinds

# def remove_inds
# def importance_sample(xdata, n_sample, uniform_frac=None):
#     pass

def training_run(xtrain, ytr_label, sample_weights, tr_sfmx, xtest, yte_label, te_sfmx, PLOT = False, BUDGET = 5000,
                 MC_ITERS = 10, n_seed = 250, n_query = 100, nplots = 0, uniform_frac=0.):
    import tqdm
    active_accs = []
    active_epes = []
    active_label_xents = []
    active_teacher_xents = []

    for MC in tqdm.tqdm(range(MC_ITERS)):
        sampling_weights = np.copy(sample_weights).ravel()
        
        xseed, seed_inds = uniform_sample(xdata=xtrain, nsample=n_seed)
        selected_inds = seed_inds

        # unlabled_inds = np.delete(np.arange(len(xtrain)), selected_inds)


        active_sampsize = []
        active_acc = []
        active_epe = []
        active_label_xent = []
        active_teacher_xent = []

        while len(selected_inds) < BUDGET:
            # print(len(selected_inds))

            # unlabled_inds = np.delete(np.arange(len(xtrain)), selected_inds)
            # xunlabel = xtrain[unlabled_inds]
            # ytr_unlabled = ytr_label[unlabled_inds]
            # tr_full_posterior = tr_sfmx[unlabled_inds]

            xselected = xtrain[selected_inds]
            yselected = ytr_label[selected_inds]
            # posterior_selected = tr_sfmx[selected_inds]

            xstage1 = xselected
            ystage1 = yselected
            # posterior_stage1 = posterior_selected

            # xtrain, tr_posterior, ytr_label = generate_samples(n_samples=N)
            mu = np.mean(xstage1, axis=0)
            std = np.std(xstage1, axis=0)
        #     mu_std.append([mu, std])
            _xstage1 = (xstage1 - mu)/std
            classifier = train_classifier(xtrain=_xstage1, ytrain=ystage1, classifier='mlp')
            # _xstage1
            _tr_acc = evaluate_classifier(classifier, _xstage1, ystage1, accuracy)
            _tr_mse = evaluate_classifier(classifier, _xstage1, ystage1, mse)

            _xtest = (xtest - mu)/std
            test_epe =  evaluate_classifier(classifier, _xtest, yte_label, accuracy)
            test_acc =  evaluate_classifier(classifier, _xtest, yte_label, mse)
            test_xent = np.mean(evaluate_classifier(classifier, _xtest, yte_label, xentropy))
            
            test_preds = classifier.predict_proba(_xtest)
            # print('Test preds shape', test_preds.shape)
            # print('Test softmax shape', te_sfmx.shape)
            # print(test_preds[:5,:])
            # print(np.flip(te_sfmx[:5,:], axis=1))
            xents = calc_crossentropy(te_sfmx, test_preds+1e-20)
            # print(np.shape(xents))
            test_teacher_xent = np.mean(xents)
            
            active_sampsize.append(len(selected_inds))
            active_acc.append(1-test_acc)
            active_epe.append(test_epe)
            active_label_xent.append(test_xent)
            active_teacher_xent.append(test_teacher_xent)
#             print('-----------------------------------------------------------------------------')
#             print('%d data samples, queried: %d, total %d ' % (len(selected_inds)-n_query, n_query, len(selected_inds)))
#             print('EPE ', test_epe, '\tacc: ', 1-test_acc, '\txent: ', test_xent)
#             print(test_teacher_xent)


            """
            Select Samples (used in the next iteration)
            """


            # pred_logits = classifier.predict_proba(_xstage1)
    #         proba_noise = np.random.randn()
#             min_logit_value_inds = argmin_logit_uncertainty_rank(xdata=xtrain, predicted_probas=pred_logits, randomize=False,
#                                                                        top_k=n_query, ignore_inds=selected_inds, uniform_frac=.15)

            newly_chosen_inds = weight_based_sample(sampling_distribution=sampling_weights, top_k=n_query,
                                                    ignore_inds=selected_inds, uniform_frac=uniform_frac)
            # print(len(selected_inds))
            # print(len(xents))
            if nplots<nplots*3 and PLOT:
                plt.figure(figsize=(18, 4))
                plt.subplot(1, 3, 1)
                # print(xtrain.shape)
                plt.scatter(xtrain[selected_inds, 0], xtrain[selected_inds, 1], marker='.',
                # plt.scatter(_xstage1[:, 0], _xstage1[:, 1], marker='.',
                #             c=classifier.predict_proba(_xstage1)[:, 1], vmin=0., vmax=1.)
                            c=classifier.predict_proba(_xstage1)[:, 1], vmin=0., vmax=1.)
    #             plt.scatter(xtrain[min_logit_value_inds, 0], xtrain[min_logit_value_inds, 1],
    #                         c=np.ones_like(min_logit_value_inds), cmap='Greys', vmin=0., vmax=1., label='Selected')
                plt.grid()
    #             plt.colorbar()
    #             plt.colorbar(matplotlib.cm.ScalarMappable(cmap='hsv'))
                plt.colorbar()
                # print(np.min(_xstage1))
                # print(np.max(_xstage1))
                plt.title('Learned Train Posterior, %d Samp Size' % len(selected_inds))
                plt.ylim(-4.25, 4.25)
                plt.xlim(-4.25, 4.24)
    #             plt.ylim(-2, 0)
    #             plt.xlim(-2, 0)
    #             plt.figure()
                plt.subplot(1, 3, 2)
    #             plt.scatter(xtrain[selected_inds, 0], xtrain[selected_inds, 1],
    #                         c=classifier.predict_proba(_xtrain)[selected_inds, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
    #             plt.scatter(xtrain[:, 0], xtrain[:, 1], marker='.',
    #                         c=classifier.predict_proba(_xtrain)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
    #                         c=classifier.predict_proba(_xtrain)[:, 1], vmin=0., vmax=1., label='Train')
    #             wts = classifier.predict_proba(_xtrain)[:, 1]
    #             inds = np.argwhere(np.abs(wts-.5)<.1)
    #             print(len(inds))
                plt.scatter(xtest[:, 0], xtest[:, 1],
                            c=classifier.predict_proba(_xtest)[:, 1], vmin=0., vmax=1., label='Test')
    #                         c=classifier.predict_proba(_xtest)[:, 1], vmin=0., vmax=1., label='Test')

    #             plt.scatter(xtrain[min_logit_value_inds, 0], xtrain[min_logit_value_inds, 1],
    #                         c=np.ones_like(min_logit_value_inds), cmap='Greys', vmin=0., vmax=1., label='Selected')
                plt.legend()
                plt.grid()
    #             plt.colorbar(matplotlib.cm.ScalarMappable(cmap='hsv'))
                plt.title('Prediction on Test Data')
                plt.colorbar()
                plt.ylim(-4.25, 4.25)
                plt.xlim(-4.25, 4.24)
    #             plt.ylim(-2, 0)
    #             plt.xlim(-2, 0)
                plt.subplot(1, 3, 3)
    #             plt.scatter(xtrain[selected_inds, 0], xtrain[selected_inds, 1],
    #                         c=classifier.predict_proba(_xtrain)[selected_inds, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
    #             plt.scatter(xtrain[:, 0], xtrain[:, 1],
    #                         c=classifier.predict_proba(_xtrain)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Train')
    #             plt.scatter(xtest[:, 0], xtest[:, 1],
    #                         c=classifier.predict_proba(_xtest)[:, 1], cmap='hsv', vmin=0., vmax=1., label='Test')

    #             pred_logits = classifier.predict_proba(_xtrain)
    #             probainds = np.argmax(pred_logits, axis=1)
    #             largest_logit = pred_logits[np.arange(len(pred_logits)), probainds]
    #             sortinds = np.argsort(largest_logit)
    #             print(len(sortinds))
    #             sortinds = np.delete(sortinds, selected_inds)
    #             print(len(sortinds))
    #             if ignore_inds is not None:
    #                 sortinds = np.delete(sortinds, ignore_inds)
    #             inds = sortinds[:n_query]

                plt.scatter(xtrain[newly_chosen_inds, 0], xtrain[newly_chosen_inds, 1], marker='.',
                            c=np.ones_like(newly_chosen_inds), cmap='Greys', vmin=0., vmax=1., label='Selected')
                plt.legend()
                plt.grid()
    #             plt.colorbar(matplotlib.cm.ScalarMappable(cmap='hsv'))
                plt.title('Selected Points')
                plt.colorbar()
                plt.ylim(-4.25, 4.25)
                plt.xlim(-4.25, 4.24)
    #             plt.ylim(-2, 0)
    #             plt.xlim(-2, 0)
                plt.show()
                nplots+=1
    #         if nplots>10:
    #             print('asdf')
    #             print(jkj094)
            selected_inds = np.concatenate([selected_inds, newly_chosen_inds])


        active_epes.append(active_epe)
        active_accs.append(active_acc)
        active_label_xents.append(active_label_xent)
        active_teacher_xents.append(active_teacher_xent)
    # print('Xent w.r.t oracle:', active_teacher_xents)
    return active_accs, active_epes, active_label_xents, active_teacher_xents