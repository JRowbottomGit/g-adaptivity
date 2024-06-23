import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GlobalFeatureExtractorCNN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dim=2, num_layers=4):
        super(GlobalFeatureExtractorCNN, self).__init__()

        if dim == 1:
            conv_fct = nn.Conv1d
        elif dim == 2:
            conv_fct = nn.Conv2d

        self.convs = nn.ModuleList([conv_fct(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)])

        for _ in range(num_layers - 2):
            self.convs.append(conv_fct(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1))

        self.convs.append(conv_fct(mid_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if dim == 1:
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, u):
        # Normalization by maximum absolute value
        u = u / torch.max(torch.abs(u))
        for conv in self.convs:
            u = F.selu(conv(u))
        u = self.global_avg_pool(u)
        u = u.view(u.size(0), -1)  # Flatten the tensor
        return u


class GlobalFeatureExtractorGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalFeatureExtractorGNN, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels)
        self.conv2 = GATConv(out_channels, out_channels)

    def forward(self, u, edge_index, batch=None):
        u = F.selu(self.conv1(u, edge_index))
        u = F.selu(self.conv2(u, edge_index))
        u = global_mean_pool(u, batch)  # Global pooling
        return u