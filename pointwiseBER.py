import numpy as np 
import os 
import math
import pdb
import random
import time
import multiprocessing as mp
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn import mixture 
from joblib import  Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import scipy.io as sio



def BinaryKernelBayesLowerbound(Data, Sigma = 0.2):
	"""
	Graph construction and connectional construction
	"""
	NumCores = mp.cpu_count()
	# # weight matrix
# 	pdb.set_trace()
	TimeStart = time.clock()
	GraphMatrix = pairwise_distances(Data[:, :-1], n_jobs = NumCores)
	ConnectMatrix = pairwise_distances(Data[:, -1].reshape((-1, 1)), n_jobs = NumCores)

	"""
	Lowerbound estimation
	"""
	N = len(GraphMatrix)
	GraphMask = GraphMatrix < np.mean(GraphMatrix) * Sigma
	PointBayesLowerbound = []
	"""
	ParallelProcessing
	"""
	def NonparEstimate(i):
		# print(i)
		kernelData = Data[GraphMask[i], :]
		KernelGraphMatrix = GraphMatrix[GraphMask[i], :][:, GraphMask[i]]
		KernelConnectMatrix = ConnectMatrix[GraphMask[i], :][:, GraphMask[i]]		
		# cluster KNN error calculation by minimum spanning tree
		Tcsr = minimum_spanning_tree(csr_matrix(KernelGraphMatrix))	
		minimum_tree = Tcsr.toarray()
		# pdb.set_trace()
		NumFR = np.sum(minimum_tree.astype(bool) * KernelConnectMatrix) # Binary case

		# Binary case
		N0 = np.sum(kernelData[:, -1] == 0)
		N1 = np.sum(kernelData[:, -1] == 1)
		# print("Point %d Finished"%i)
		return 0.5 - 0.5*math.sqrt(max(0, 1 - 2 * NumFR / (N0+N1)))	

	
	PointBayesLowerbound = Parallel(n_jobs=NumCores)(delayed(NonparEstimate)(i) for i in range(N))

	
	TimeElapsed = (time.clock()-TimeStart)
	print(TimeElapsed)
	return PointBayesLowerbound

def main():
	"""
	Synthetic data loading
	"""
	# LoadDir =  '/home/weizhi/Desktop/Research/SoftReg/Results/SynResults/stat/embedding/'
	# SaveDir = '/home/weizhi/Desktop/Research/SoftReg/Results/SynResults/stat/estimation/Nonparametric/'
	# Data = np.load(LoadDir + 'data_longfeat_2_pca_train.npy')
	
	"""
	Higgs data loading
	"""
	LoadDir =  '/home/weizhi/Desktop/Research/SoftReg/Results/RealResults/Higgs/stat/embedding/'
	SaveDir = '/home/weizhi/Desktop/Research/SoftReg/Results/RealResults/Higgs/stat/estimation/Nonparametric/PointSLS/'
	Data = np.load(LoadDir + 'PCASampleTrainHiggs.npy')
	Sigma = 0.2
	# pdb.set_trace()
	PointBER = BinaryKernelBayesLowerbound(Data)
# 	pdb.set_trace()
	np.save(SaveDir + 'PointBERSampleTrainHiggs.npy', PointBER)


from sklearn.metrics import brier_score_loss
if __name__=="__main__":
	main()
