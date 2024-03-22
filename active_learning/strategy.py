import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, MLP
from torch.nn import Linear


class Strategy:
    def __init__(self, dataset, model, criterion, optimizer, device):
        self.dataset = dataset # this is a `Dataset` object
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    # implemented differently in each strategy
    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, nb_epoch):
        labeled_idxs = self.dataset.get_labeled_data()
        print("Starting training loop....")
        # Train the model
        for epoch in range(1, nb_epoch):
            self.model.train()
            total_loss = 0
            for i, data in enumerate(self.dataset.train):
                #print("Train data shape: ", data.x.shape)
                if i not in labeled_idxs:
                    continue
                data = data.to(self.device)
                #print(data.shape)
                self.optimizer.zero_grad()
                output = self.model(data)#.x, data.edge_index, data.batch)
                #print("Output shape: ", output.shape)
                # print(output[0])
                # print(data.y[0])
                # print("\n\n\n")
                #print("Output shape: ")
                #print(output.shape)
                #output = global_mean_pool(output, data.batch)
                loss = self.criterion(output, data.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            #print(len(labeled_idxs))
            epoch_loss = total_loss / len(labeled_idxs)
            print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}')

    def eval(self, test_data):
        total_loss = 0
        self.model.eval()
        print("Evaluating model on " + str(len(test_data)) + " batches...")
        for data in test_data:
            data = data.to(self.device)
            output = self.model(data)#.x, data.edge_index, data.batch)
            #output = global_mean_pool(output, data.batch)
            loss = self.criterion(output, data.y)
            total_loss += loss.item()

        return total_loss/len(test_data)

    def predict_with_uncertainty(self, candidate_idxs, n_iterations=3):
        self.model.train()  # Enable dropout
        means = []
        stds = []

        print("Predicting with uncertainty on " + str(len(candidate_idxs)) + " batches...")
        
        for i, data in enumerate(self.dataset.train):
            if i not in candidate_idxs:
                continue
            # print("Predicting with uncertainty on batch " + str(i) + "...")
            data = data.to(self.device)
            predictions = [self.model(data).detach() for _ in range(n_iterations)]
            # print("Length of predictions: ", len(predictions))
            # print("Shape of each prediction: ", predictions[0].shape)
            predictions = torch.stack(predictions)
            #print(predictions.shape)
            mean_predictions = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)
            # print(uncertainty)
            # print("Mean predictions: ", mean_predictions.shape)
            #print("Uncertainty: ", uncertainty.shape)
            # print("Uncertainty value: ", uncertainty.mean(dim=0).item())
            stds.append((i, uncertainty.mean(0).mean(0).item()))

        self.model.eval()

        # print("Length of stds: ", len(stds))
        return mean_predictions, stds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings
