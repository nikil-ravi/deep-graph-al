import numpy as np
from strategy import Strategy

class MCDropout(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        _, uncertainty = self.predict_with_uncertainty(self.model, unlabeled_idxs)
        uncertainty_sorted, idxs = uncertainty.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]