
import numpy as np
from torch.utils.data import DataLoader, Subset

class Data:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        
        self.n_pool = len(train)
        self.n_test = len(test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        # print("Initial Labels: ", tmp_idxs[:num])
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        print("Number of labels: ", np.count_nonzero(labeled_idxs))
        return labeled_idxs #, Subset(self.train, list(labeled_idxs))
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, Subset(self.train, list(unlabeled_idxs))
    
    def cal_test_acc(self, preds):
        deepchem.metrics.Metric(metrics.pearson_r2_score,
                                            np.mean)
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def transforms():
        pass