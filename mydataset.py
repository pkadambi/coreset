from sklearn.model_selection import train_test_split


class MyDataset:
    def __init__(self, x_data, y_data, tr_frac, cal_frac, te_frac):
        '''
        xdata - full dataset (with train and test)
        ydata - full labels 
        tr_frac - fraction of datset to be in test set
        cal_frac - fraction of dataset to be in calibration set
        te_frac - fraction of dataset to be in test set
        '''
        self.xdata = x_data
        self.ydata = y_data        
        self.gen_splits(train_fra)
    
    def gen_splits(self, train_frac = .7, calib_frac = .1, test_frac = .2):
        #TODO: this should be part of the init function somehow, since this kind of thing should be handled when the datset is created
        
        assert train_frac + calib_frac + test_frac == 1., 'Sum of train/test/calib split must equal 1'
        
        self.train_frac, self.test_frac, self.calib_frac = train_frac, test_frac, calib_frac

        inds = np.arange(len(self.xdata))
        xtrcalib, xtest, ytrcalib, ytest, inds_trcalib, inds_test = train_test_split(self.xdata, self.ydata, inds, test_size=test_frac)
        xtrain, xcalib, ytrain, ycalib, inds_tr, inds_calib = train_test_split(xtrcalib, ytrcalib, inds_trcalib, test_size=calib_frac/(1-test_frac))
        
        self.xtrain, self.ytrain = xtrain, ytrain
        self.xtest, self.ytest = xtest, ytest
        self.xcalib, self.ycalib = xcalib, ycalib
        
        self.inds_trcalib = inds_trcalib
        self.inds_tr = inds_tr
        self.inds_test = inds_test
        self.inds_calib = inds_calib
        print('Split dataset. n_train %d, n_calib %d, n_test %d.' % (len(xtrain), len(xcalib), len(xtest)))

