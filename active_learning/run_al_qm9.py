import os
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
from models import GAT, GINRegression
from al_data import Data
from random_strategy import RandomSampling
import argparse
from mc_dropout import MCDropout



TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
    "h298", "g298" # TODO: need to be 7 more total, try to find from paper and add, if not it's fine as I don't use this anywhere
]


def main():
    parser = argparse.ArgumentParser(description='Active Learning experiment arguments.')

    parser.add_argument('--dataset', type=str, help='QM9', required=False, default="QM9")
    parser.add_argument('--num_tasks', type=int, help='Number of tasks/classes for regression/classification', required=False, default=5)
    parser.add_argument('--num_rounds', type=int, help="Number of rounds for active learning", required=False, default=0)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training", required=False, default=30)
    parser.add_argument('--init_labeled', type=int, help="Percentage of initial labeled data points", required=False, default=10)
    parser.add_argument('--strategy', type=str, help="Active learning strategy to use", required=False, default="random")
    parser.add_argument('--model', type=str, help="Model", required=False, default="GIN")
    parser.add_argument('--query_size', type=int, help="Percent of data points to query", required=False, default=10)
    parser.add_argument('--save_path', type=str, help="Folder to save the models and logs", required=False, default="./results") #TODO: use date, time here

    args = parser.parse_args()

    print("Arguments: ", args)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == "QM9":
        dataset = QM9(root= '../QM9', transform=NormalizeFeatures())
        dataset.data.y = dataset.data.y[:, :args.num_tasks]
        train_dataset = dataset[:100000]
        test_dataset = dataset[100000:120000]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        data = Data(train_loader, test_loader)
        criterion = nn.MSELoss(reduction='mean')

    if args.model == "GIN":
        model = GINRegression(dataset.num_node_features, out_channels=args.num_tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    #strategy
    if args.strategy == "random":
        strategy = RandomSampling(data, model, criterion, optimizer, device)
    elif args.strategy == "mcdropout":
        strategy = MCDropout(data, model, criterion, optimizer, device)
        
    #initial labeled data
    data.initialize_labels(num=int(len(train_loader)*args.init_labeled/100))

    print("My data object: ", data)
    print("Model: ", model)
    print("Strategy: ", strategy)
    print("Size of initial training dataset: ", len(train_loader))
    print("Initial label IDs created")
    print("Beginning training/active learning")
    print("______________________________________________________________________________________")

    # First round
    strategy.train(args.num_epochs)
    preds = strategy.predict(data.test)
    print("0th round accuracy:", preds)

    # Save the model
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    torch.save(model.state_dict(), args.save_path + "/round_0.pth")

    for r in range(1, args.num_rounds + 1):

        print("Round: ", r)

        # Query the points we want labels for
        query_idxs = strategy.query(n_query=int(len(train_loader)*args.query_size/100))

        # update available labels
        strategy.update(query_idxs)
        
        # reset the net
        print("Resetting the net and optimizer")
        strategy.model.reset_parameters() #GIN(in_channels=dataset.num_node_features, hidden_channels=32, num_layers=5, out_channels=19).to(device)
        strategy.optimizer = torch.optim.Adam(strategy.model.parameters(), lr=0.005)

        strategy.train(args.num_epochs)

        # calculate accuracy
        preds = strategy.predict(data.test)

        #print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
        print(f"Round {r} testing accuracy: {preds}")

        # Save the model
        torch.save(model.state_dict(), args.save_path + "/round_1.pth")

if __name__ == "__main__":
    main()
