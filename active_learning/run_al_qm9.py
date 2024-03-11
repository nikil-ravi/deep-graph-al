import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn.models import GIN
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, global_mean_pool
from models import GAT
from al_data import Data
from random_strategy import RandomSampling

dataset = QM9(root= '../QM9')
dataset.transform = NormalizeFeatures()

#dataset.data.y = dataset.data.y[:, :5]

train_dataset = dataset[:100000]
test_dataset = dataset[100000:120000]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#data = Data(train_loader, test_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GAT(in_channels=dataset.num_node_features, hidden_channels=32, out_channels=5).to(device)
model = GIN(in_channels=dataset.num_node_features, hidden_channels=32, num_layers=5, out_channels=19).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss(reduction='mean')

print("Model: ", model)
print("Device: ", device)
print("Optimizer: ", optimizer)
NUM_ROUNDS = 0
NUM_EPOCHS = 5
DATASET = 'qm9'
TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
    "h298", "g298" # TODO: need to be 7 more total, try to find from paper and add, if not it's fine as I don't use this anywhere
]


data = Data(train_loader, test_loader)

print("My data object: ", data)
strategy = RandomSampling(data, model, criterion, optimizer, device)

print("Strategy: ", strategy)

print("Size of initial training dataset: ", len(train_loader))

data.initialize_labels(num=int(len(train_loader)/5))

print("Initial label IDs created")

print("Beginning training/active learning")
print("_______________________________________________________")

# First round

strategy.train(NUM_EPOCHS)
preds = strategy.predict(data.test)
print("0th round accuracy:", preds)

for r in range(1, NUM_ROUNDS + 1):

    print("Round: ", r)

    # Query the points we want labels for
    query_idxs = strategy.query(int(len(train_loader)/10))

    # update labels
    strategy.update(query_idxs)
    # reset the net
    print("Resetting the net and optimizer")
    #strategy.model = GAT(in_channels=dataset.num_node_features, hidden_channels=32, out_channels=5).to(device)
    strategy.model.reset_parameters() #GIN(in_channels=dataset.num_node_features, hidden_channels=32, num_layers=5, out_channels=19).to(device)
    strategy.optimizer = torch.optim.Adam(strategy.model.parameters(), lr=0.005)
    print("Model and optimizer: ")
    print(strategy.model)
    print(strategy.optimizer)
    strategy.train(NUM_EPOCHS)

    # calculate accuracy
    preds = strategy.predict(data.test)
    #print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
    print(f"Round {r} testing accuracy: {preds}")
