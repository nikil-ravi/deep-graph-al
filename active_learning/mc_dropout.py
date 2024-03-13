import numpy as np
from strategy import Strategy

class MCDropout(Strategy):
    def __init__(self, dataset, model, criterion, optimizer, device):
        super(MCDropout, self).__init__(dataset, model, criterion, optimizer, device)

    def query(self, n_query, n_iterations=10):
        unlabeled_idxs = self.dataset.get_unlabeled_data()
        print("Unlabeled indices: ", len(unlabeled_idxs))
        _, batch_uncertainties = self.predict_with_uncertainty(unlabeled_idxs, n_iterations=n_iterations)
        # print(batch_uncertainties)
        sorted_batches = sorted(batch_uncertainties, key=lambda x: x[1], reverse=True) # sort in descending order
        # print(sorted_batches)
        indices_to_label = [batch_uncertainty[0] for batch_uncertainty in sorted_batches[:n_query]]
        print("Length of selected indices to query: ", len(indices_to_label))
        # print("Selected indices to query: ", indices_to_label)
        return indices_to_label