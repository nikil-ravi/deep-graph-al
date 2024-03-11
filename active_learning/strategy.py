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
                if i not in labeled_idxs:
                    continue
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data.x, data.edge_index, data.batch)
                # print(output[0])
                # print(data.y[0])
                # print("\n\n\n")
                print("Output shape: ")
                print(output.shape)
                loss = self.criterion(output, data.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(labeled_idxs)
            print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}')

    def predict(self, test_data):
        total_loss = 0
        self.model.eval()
        print("Evaluating model on " + str(len(test_data)) + " batches...")
        for data in test_data:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(output, data.y)
            total_loss += loss.item()

        return total_loss/len(test_data)

    def predict_with_uncertainty(self, model, input_data, n_iterations=100):
        model.train()  # Enable dropout
        predictions = [self.model(input_data) for _ in range(n_iterations)]
        predictions = torch.stack(predictions)
        mean_predictions = predictions.mean(0)
        uncertainty = predictions.std(0)
        return mean_predictions, uncertainty

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
