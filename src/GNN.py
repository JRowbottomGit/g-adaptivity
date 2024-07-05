import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.utils import add_self_loops, remove_self_loops

from firedrake_difFEM.difFEM_1d import torch_FEM_1D
from firedrake_difFEM.difFEM_2d import torch_FEM_2D
from params import get_arg_list
from feature_extractors import GlobalFeatureExtractorGNN, GlobalFeatureExtractorCNN
from utils_data import reshape_grid_to_fd_tensor, reshape_fd_tensor_to_grid
from GRAND_plus import GRAND_conv, GRAND_plusConv, GAT_plus

class MLP(torch.nn.Module):
    def __init__(self, dataset, opt):
        super().__init__()
        self.opt = opt
        in_dim = dataset.data.x_comp.shape[1]
        hid_dim = opt['hidden_dim']
        out_dim = dataset.data.x_comp.shape[1]
        self.enc = get_enc(opt, in_dim, hid_dim, nonlin_type=self.opt['non_lin'])
        self.non_lin = get_nonlin(self.opt['non_lin'])
        self.dec = get_dec(opt, hid_dim, out_dim, nonlin_type=self.opt['non_lin'])
        self.fc1 = torch.nn.Linear(hid_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)

    def forward(self, data):
        x = data.x_comp
        x = self.enc(x)
        if self.opt['residual']:
            x = x + self.opt['time_step'] * self.fc1(x)
        else:
            x = self.fc1(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.non_lin(x)
        if self.opt['residual']:
            x = x + self.opt['time_step'] * self.fc2(x)
        else:
            x = self.fc2(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.non_lin(x)
        x = self.dec(x)
        return x


def get_nonlin(nonlin_type):
    if nonlin_type == 'relu':
        return nn.ReLU()
    elif nonlin_type == 'elu':
        return nn.ELU()
    elif nonlin_type == 'selu':
        return nn.SELU()
    elif nonlin_type == 'tanh':
        return nn.Tanh()
    elif nonlin_type == 'sigmoid':
        return nn.Sigmoid()
    elif nonlin_type == 'leaky_relu':
        return nn.LeakyReLU()
    elif nonlin_type == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError

def get_mlp(in_dim, hid_dim, out_dim, nonlin_type):
    non_lin = get_nonlin(nonlin_type)
    #return 2 layer MLP with activation
    return nn.Sequential(nn.Linear(in_dim, hid_dim), non_lin,
                        nn.Linear(hid_dim, out_dim), non_lin)

def get_enc(opt, in_dim, out_dim, hid_dim=None, nonlin_type='relu'):
    encdec_type = opt['enc']

    if encdec_type == 'identity':
        if out_dim >= in_dim:
            #return identity function in the first in_dim dimensions and 0 in the rest
            id_mat = torch.zeros(in_dim, out_dim)
            id_mat[:in_dim, :in_dim] = torch.eye(in_dim)
            id_lin = nn.Linear(in_dim, out_dim, bias=False)
            id_lin.weight.data = id_mat.T
            id_lin.weight.requires_grad = False
        else:
            # return identity function in the first out_dim dimensions drop the rest
            id_mat = torch.zeros(out_dim, in_dim)
            id_mat[:out_dim, :out_dim] = torch.eye(out_dim)
            id_lin = nn.Linear(in_dim, out_dim, bias=False)
            id_lin.weight.data = id_mat
            id_lin.weight.requires_grad = False
        return id_lin
    elif encdec_type == 'lin_layer':
        return nn.Linear(in_dim, out_dim)
    elif encdec_type == 'MLP':
        if hid_dim is None:
            hid_dim = in_dim
        return get_mlp(in_dim, hid_dim, out_dim, nonlin_type=nonlin_type)
    else:
        raise NotImplementedError


def get_dec(opt, in_dim, out_dim, hid_dim=None, nonlin_type='relu'):
    encdec_type = opt['enc']
    if encdec_type == 'identity':
        id_lin = nn.Identity(in_dim, out_dim)
        return id_lin


def get_conv(opt, conv_type, in_dim, out_dim, feat_dim=None):
    if conv_type == 'GCN':
        return GCNConv(in_dim, out_dim)
    elif conv_type == 'GAT':
        return GATConv(in_dim, out_dim, heads=1)
    elif conv_type == 'TRANS':
        return TransformerConv(in_dim, out_dim, heads=1)
    elif conv_type == 'GRAND':
        return GRAND_conv(opt, in_dim, out_dim, heads=1)
    elif conv_type == 'GRAND_plus':
        return GRAND_plusConv(opt, in_dim, out_dim, global_feat_dim=feat_dim, heads=1, concat=False, beta=False, dropout=0.0,
                                         edge_dim=None, bias=False, root_weight=False)
    elif conv_type == "GAT_plus":
        return GAT_plus(opt, in_dim, out_dim)
    else:
        # todo laplacian diffusion with learnable diffusivity / grad dependent monitor function / uncertainty
        raise NotImplementedError


def build_conv_list(opt):
    layers = []

    #conv
    if opt['share_conv']:
        share_conv = get_conv(opt, opt['conv_type'], opt['hidden_dim'], opt['hidden_dim'], opt['global_feat_dim'])

    for i in range(opt['num_layers']):
        # for each layer, add conv, nonlin, dropout
        if opt['share_conv']:
            conv = share_conv
        else:
            conv = get_conv(opt, opt['conv_type'], opt['hidden_dim'], opt['hidden_dim'], opt['global_feat_dim'])
        layers.append(conv)
    return nn.ModuleList(layers)


class GNN(nn.Module):
    def __init__(self, dataset, opt):
        super(GNN, self).__init__()
        self.dataset = dataset
        self.opt = opt
        self.dim = dataset.num_x_comp_features
        self.mesh_dims = get_arg_list(opt['mesh_dims'])
        self.in_dims = [self.dim]
        if self.opt['gnn_inc_feat_f']:
            self.in_dims += [1]
        if self.opt['gnn_inc_feat_uu']:
            self.in_dims += [1]
        if self.opt['gnn_inc_glob_feat_f']:
            self.in_dims += [opt['global_feat_dim']]
        if self.opt['gnn_inc_glob_feat_uu']:
            self.in_dims += [opt['global_feat_dim']]

        opt['hidden_dims_list'] = self.in_dims

        in_dim = sum(self.in_dims)
        hid_dim = opt['hidden_dim']
        out_dim = self.dim

        self.enc = get_enc(opt, in_dim, hid_dim, nonlin_type=self.opt['non_lin'])
        self.conv_layers = build_conv_list(opt)
        self.non_lin = get_nonlin(self.opt['non_lin'])
        self.dec = get_dec(opt, hid_dim, out_dim, nonlin_type=self.opt['non_lin'])

        if self.opt['gnn_inc_glob_feat_f']:
            self.global_out_dim = opt['global_feat_dim']
            self.global_feature_extractor_cnn_f = GlobalFeatureExtractorCNN(1, hid_dim, self.global_out_dim, dim=self.dim) #opt['in_channels'], opt['out_channels'])
        if self.opt['gnn_inc_glob_feat_uu']:
            self.global_out_dim = opt['global_feat_dim']
            self.global_feature_extractor_cnn_uu = GlobalFeatureExtractorCNN(1, hid_dim, self.global_out_dim, dim=self.dim) #opt['in_channels'], opt['out_channels'])

        if self.opt['learn_step']:
            self.steps = nn.ParameterList([nn.Parameter(torch.tensor([self.opt['time_step']])) for _ in range(opt['num_layers'])])

        if self.dim == 1:
            self.quad_points = torch.linspace(0, 1, self.opt['eval_quad_points'])
        elif self.dim == 2:
            x0 = torch.linspace(0, 1, self.opt['eval_quad_points'])  # , dtype=torch.float64)
            y0 = torch.linspace(0, 1, self.opt['eval_quad_points'])  # , dtype=torch.float64)
            [X, Y] = torch.meshgrid(x0, y0, indexing='ij')
            self.quad_points = [X, Y]

    def forward(self, data):
        batch = data.batch
        batch_size = data.batch.max().item() + 1
        num_in_batch = batch.max().item() + 1
        edge_index = data.edge_index.to(self.opt['device'])
        x_comp = data.x_comp.to(self.opt['device'])
        f = data.f_tensor.to(self.opt['device'])
        uu = data.uu_tensor.to(self.opt['device'])
        u_true = data.u_true_tensor.to(self.opt['device'])
        if self.opt['data_type'] == 'randg_mix':# and hasattr(data, 'batch_dict'):
            pde_params = {}
            pde_params['centers'] = {i: data.batch_dict[i]['pde_params']['centers'] for i in range(batch_size)}
            pde_params['scales'] = {i: data.batch_dict[i]['pde_params']['scales'] for i in range(batch_size)}
        else:
            pde_params = data.pde_params

        if self.opt['fix_boundary']:
            mask = ~data.to_boundary_edge_mask * ~data.to_corner_nodes_mask * ~data.diff_boundary_edges_mask
            edge_index = edge_index[:, mask]
            if self.dim == 1:
                corner_nodes = torch.cat([torch.tensor([0 + b * self.opt['mesh_dims'][0], (1 + b) * self.opt['mesh_dims'][0] - 1]) for b in range(num_in_batch)]).repeat(2, 1)
                edge_index = torch.cat([edge_index, corner_nodes], dim=1)
            elif self.dim == 2:
                corner_nodes = torch.stack([torch.from_numpy(arr).to(self.opt['device']) for arr in data.corner_nodes]) #batch x 4
                num_each_nodes = batch.unique(return_counts=True)[1]
                cum_num_each_nodes = torch.cumsum(num_each_nodes,dim=0).to(self.opt['device'])
                corner_nodes[1:] += cum_num_each_nodes[:-1].unsqueeze(-1)
                corner_edges = corner_nodes.reshape(-1).repeat(2,1).to(self.opt['device'])
                edge_index = torch.cat([edge_index, corner_edges], dim=1)

        if self.opt['self_loops']:
            num_nodes = x_comp.size(0)
            edge_index, edge_attr = remove_self_loops(edge_index)
            edge_index, edge_attr = add_self_loops(edge_index, num_nodes=num_nodes)

        if self.dim == 1:
            x_comp = x_comp.unsqueeze(-1)

        features = x_comp
        feature_names = ['x_comp']
        if self.opt['gnn_inc_feat_f']:
            if self.opt['gnn_normalize']:
                f = f / torch.max(f)
            features = torch.cat([features, f.unsqueeze(-1)], dim=1)
            feature_names.append('f')
        if self.opt['gnn_inc_feat_uu']:
            if self.opt['gnn_normalize']:
                uu = uu / torch.max(uu)
            features = torch.cat([features, uu.unsqueeze(-1)], dim=1)
            feature_names.append('uu')

        #Global features
        if self.opt['gnn_inc_glob_feat_f']:
            if self.opt['data_type'] != 'randg_mix':
                num_nodes = self.dataset.x_comp_shared.shape[0]
                f_grid = reshape_fd_tensor_to_grid(f, self.dataset.mapping_tensor, [num_nodes, num_nodes], batch_size, self.dim)
            else:
                num_nodes = int(np.sqrt(data.x_comp.shape[0]))
                f_grid = reshape_fd_tensor_to_grid(f, data.mapping_tensor, [num_nodes, num_nodes], batch_size, self.dim)

            global_features_cnn_f = self.global_feature_extractor_cnn_f(f_grid.unsqueeze(1))
            repeats = torch.bincount(batch)
            global_features_cnn_f_expanded = global_features_cnn_f.repeat_interleave(repeats, dim=0)
            features = torch.cat([features, global_features_cnn_f_expanded], dim=-1).float()
            feature_names.append('f_global')

        if self.opt['gnn_inc_glob_feat_uu']:
            if self.opt['data_type'] != 'randg_mix':
                num_nodes = self.dataset.x_comp_shared.shape[0]
                uu_grid = reshape_fd_tensor_to_grid(uu, self.dataset.mapping_tensor, [num_nodes, num_nodes], batch_size, self.dim)
            else:
                num_nodes = int(np.sqrt(data.x_comp.shape[0]))
                uu_grid = reshape_fd_tensor_to_grid(uu, data.mapping_tensor, [num_nodes, num_nodes], batch_size, self.dim)

            global_features_cnn_uu = self.global_feature_extractor_cnn_uu(uu_grid.unsqueeze(1))
            repeats = torch.bincount(batch)
            global_features_cnn_uu_expanded = global_features_cnn_uu.repeat_interleave(repeats, dim=0)
            features = torch.cat([features, global_features_cnn_uu_expanded], dim=-1).float()
            feature_names.append('uu_global')

        x = self.enc(features)
        x = F.dropout(x, self.opt['dropout'], training=self.training)

        for i, layer in enumerate(self.conv_layers):
            if self.opt['residual']:
                if self.opt['conv_type'] == 'GRAND_plus':

                    for b_idx in range(num_in_batch):
                        if self.opt['data_type'] == 'randg_mix':
                            mesh = data.mesh[b_idx]
                        else:
                            mesh = self.dataset.mesh
                    res = layer(x, edge_index, features, mesh)
                else:
                    res = layer(x, edge_index)
                    res = F.dropout(res, self.opt['dropout'], training=self.training)
                    res = self.non_lin(res)

                if self.opt['learn_step']:
                    x = x + self.steps[i] * res
                else:
                    x = x + self.opt['time_step'] * res

            else:
                x = layer(x, edge_index)
                x = F.dropout(x, self.opt['dropout'], training=self.training)
                x = self.non_lin(x)

        x = self.dec(x)
        x_phys = x[:, :self.dim] # assume the first dim are the coordinates
        features = x[:, self.dim:] # and the rest are the latent features
        self.end_MLmodel = time.time()

        if self.opt['loss_type'] == 'mesh_loss':
            return x_phys
        elif self.opt['loss_type'] == 'modular':
            return x_phys
        elif self.opt['loss_type'] == 'pde_loss':
            coeffs_batched_list, x_phys_batched_list, sol_batched_list = [], [], []
            for b_idx in range(num_in_batch):
                if self.opt['data_type'] == 'randg_mix':
                    c_list_torch = [torch.from_numpy(c_0).to(self.opt['device']) for c_0 in pde_params['centers'][b_idx]]
                    s_list_torch = [torch.from_numpy(s_0).to(self.opt['device']) for s_0 in pde_params['scales'][b_idx]]
                    mesh = data.mesh[b_idx]
                else:
                    c_list_torch = [torch.from_numpy(c_0).to(self.opt['device']) for c_0 in pde_params['centers'][b_idx]]
                    s_list_torch = [torch.from_numpy(s_0).to(self.opt['device']) for s_0 in pde_params['scales'][b_idx]]
                    mesh = self.dataset.mesh

                if self.dim == 1:
                    x_phys_batch = x_phys.squeeze()[data.batch == b_idx]
                    num_meshpoints = x_phys_batch.shape[0]
                    coeffs_batch, x_phys_batch, sol_batch, BC1, BC2 = torch_FEM_1D(self.opt, x_phys_batch, self.quad_points, num_meshpoints, c_list_torch, s_list_torch)
                    coeffs_batched_list.append(coeffs_batch)
                    x_phys_batched_list.append(x_phys_batch)
                    sol_batched_list.append(sol_batch)

                elif self.dim == 2:
                    x_phys_batch = x_phys.squeeze()[data.batch == b_idx]
                    num_meshpoints = int(np.sqrt(x_phys_batch.shape[0]))
                    coeffs_batch, x_phys_batch, sol_batch = torch_FEM_2D(self.opt, mesh, x_phys_batch, quad_points=self.quad_points, num_meshpoints=num_meshpoints,
                                                                            c_list=c_list_torch, s_list=s_list_torch)

                    sol_fd_batch = reshape_grid_to_fd_tensor(sol_batch.view(-1).unsqueeze(-1), self.dataset.mapping_tensor_fine)
                    coeffs_batched_list.append(coeffs_batch)
                    x_phys_batched_list.append(x_phys_batch)
                    sol_batched_list.append(sol_fd_batch.squeeze())

            coeffs_batched = torch.cat(coeffs_batched_list, dim=0)
            x_phys_batched = torch.cat(x_phys_batched_list, dim=0)
            sol_fd_batched = torch.cat(sol_batched_list, dim=0)

            return coeffs_batched, x_phys_batched, sol_fd_batched