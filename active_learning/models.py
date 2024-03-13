import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GIN, GATConv, global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        return x

class GINRegression(torch.nn.Module):
    def __init__(self, num_features, out_channels):
        super(GINRegression, self).__init__()
        self.gin = GIN(in_channels=num_features, hidden_channels=64, num_layers=5, out_channels=out_channels, dropout=0.6)
        #self.linear = torch.nn.Linear(19, out_channels)

    def forward(self, data):
        x = self.gin(data.x, data.edge_index)

        x = global_mean_pool(x, data.batch)

        #x = self.linear(x)
        return x
    
    def reset_parameters(self):
        self.gin.reset_parameters()
        #self.linear.reset_parameters()
