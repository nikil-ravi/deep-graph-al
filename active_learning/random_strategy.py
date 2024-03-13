from strategy import Strategy
import numpy as np

class RandomSampling(Strategy):
    def __init__(self, dataset, model, criterion, optimizer, device):
        super(RandomSampling, self).__init__(dataset, model, criterion, optimizer, device)

    def query(self, n_query):
        # returns randomly selected yet-unlabeled data point indices
        ids = np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n_query, replace=False)
        # print("Selected IDs to query: ", ids)
        return ids