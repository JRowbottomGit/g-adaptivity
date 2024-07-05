import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing, TransformerConv, GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax


class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='GCN', activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0,dim_size=X.size(0), reduce='mean'))

        return gg

def softmax_temperature(temperature, x, index=None, ptr=None, size_i=None):
    #higher temperature -> uniform distribution
    return softmax(x / temperature, index, ptr, size_i)


class GRAND_plusConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self, opt,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.opt = opt #store opt for accessing params
        # self.feat_in_dims = sum(opt['hidden_dims_list'])
        self.dim = len(opt['mesh_dims'])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)

        # self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.lin_value = nn.Identity(in_channels, heads * out_channels)  # Overidding channel mixing to identity

        if self.opt['softmax_temp_type'] == 'learnable_a':
            #MLP that learns global scalar
            self.sm_temp_a = nn.Parameter(torch.Tensor(1, heads, 1))
            #with sigmoid between 0.5 and 4
            # self.sm_temp_a = nn.Sequential(
            #     Linear(in_channels[1], heads, bias=False), nn.Sigmoid())
        elif self.opt['softmax_temp_type'] == 'learnable_v':
            #MLP that maps feature vector to temperature
            self.sm_temp_v = Linear(in_channels[1], heads, bias=False)
            #with sigmoid between 0.5 and 4
            # self.sm_temp_v = nn.Sequential(
            #     Linear(in_channels[1], heads * out_channels, bias=False), nn.Sigmoid())

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

        self.stored_ei = None # attribute for storing edge_index for tracking attention weights
        self.stored_alpha = None # attribute for storing attention weights for tracking


    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()

        # self.lin_value.reset_parameters() #not resetting value as it's identity
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    # def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
    #             edge_attr: OptTensor = None, return_attention_weights=None):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, global_features: OptTensor = None, mesh=None,
                edge_attr: OptTensor = None, return_attention_weights=None):

        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        self.mesh_points = x[0]
        self.mesh = mesh

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(self.opt['show_mesh_evol_plots'], bool):
            assert alpha is not None
            self.stored_ei = edge_index
            self.stored_alpha = alpha

        #need to return Xdot = res = [A()X - X]=[A_theta-I]X not just A()X
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out - x[1], (edge_index, (alpha, query, key))
            elif isinstance(edge_index, SparseTensor):
                pass
                # return out - x[1], edge_index.set_value(alpha, layout='coo')
        else:
            return out - x[1]

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, edge_index: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.opt['reg_skew'] and self.dim == 2:
            cell_node_map = self.mesh.coordinates.cell_node_map().values  # Get cell-node connectivity
            triangles_tensor = torch.stack([self.mesh_points[cell] for cell in cell_node_map]).to(self.opt['device'])  # torch.tensor(triangles, dtype=torch.float32)
            x = triangles_tensor[:, :, 0]
            y = triangles_tensor[:, :, 1]
            # Calculate the area using the determinant method
            area = 0.5 * torch.abs(x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0]) + x[:, 2] * (y[:, 0] - y[:, 1]))
            #map triangle areas to edges
            i_idx = torch.from_numpy(cell_node_map[:, 0]).type(torch.long)
            j_idx = torch.from_numpy(cell_node_map[:, 1]).type(torch.long)
            k_idx = torch.from_numpy(cell_node_map[:, 2]).type(torch.long)

            # Collect all edges and corresponding triangle areas
            tri_edges = torch.cat([
                torch.stack([i_idx, j_idx], dim=1),
                torch.stack([j_idx, k_idx], dim=1),
                torch.stack([k_idx, i_idx], dim=1)
            ])

            reverse_tri_edges = torch.cat([
                torch.stack([j_idx, i_idx], dim=1),
                torch.stack([k_idx, j_idx], dim=1),
                torch.stack([i_idx, k_idx], dim=1)
            ])

            bi_tri_edges = torch.cat([tri_edges, reverse_tri_edges])
            edge_areas = torch.cat([area, area, area, area, area, area])

            #create indexing tensor to map tri_edges to edge_index ordering using torch.where
            edge_index_area_sum = torch.zeros(edge_index.size(1)).to(self.opt['device'])
            for i, edge in enumerate(edge_index.t().tolist()):
                # todo this might not catch all triangles for bidirectional edges as firedrake ordering might not be as simplacial complex oriented
                # trick might be to add or to where to catch both directions
                edges_hits = torch.where((tri_edges[:, 0] == edge[0]) * (tri_edges[:, 1] == edge[1]))[0]
                # if has 0 triangle: pass
                if edges_hits.size(0) == 0:
                    continue
                # if has 1 triangle: assign area
                elif edges_hits.size(0) == 1:
                    edge_index_area_sum[i] += edge_areas[edges_hits[0]]
                # if has 2 triangles: assign average area
                elif edges_hits.size(0) == 2:
                    edge_index_area_sum[i] += edge_areas[edges_hits[0]] + edge_areas[edges_hits[1]]

            alpha = alpha * edge_index_area_sum.unsqueeze(-1)

        if self.opt['softmax_temp_type'] == 'fixed':
            alpha = softmax_temperature(self.opt['softmax_temp'], alpha, index, ptr, size_i)
        elif self.opt['softmax_temp_type'] == 'learnable_a':
            alpha = softmax_temperature(self.sm_temp_a.squeeze(2), alpha, index, ptr, size_i)
        elif self.opt['softmax_temp_type'] == 'learnable_v':
            alpha = softmax_temperature(self.sm_temp_v(alpha).squeeze(2), alpha, index, ptr, size_i)
        else:
            alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

# class to take in global feature vector and output QK matrices
class QKNet(nn.Module):
    def __init__(self, opt, in_dim, out_dim):
        super(QKNet, self).__init__()
        self.opt = opt
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        Q = self.fc1(x)
        K = self.fc2(x)
        return Q, K


# simple grand conv implementation from transformer conv
class GRAND_conv(TransformerConv):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html#torch_geometric.nn.conv.TransformerConv
    def __init__(self, opt, in_channels, out_channels, heads=1, concat=False, beta=False, dropout=0,
                 edge_dim=None, bias=False, root_weight=False):
        # def __init__(self, in_channels, out_channels, dim=2, heads=1, concat=False, beta=False, dropout=0, edge_dim=None, bias=False, root_weight=False):
        super(GRAND_conv, self).__init__(in_channels, out_channels, heads=1, concat=False, beta=False, dropout=0.0,
                                         edge_dim=None, bias=False, root_weight=False) #nb these are non default values

        self.opt = opt
        # self.lin_skip = nn.Identity(in_channels, heads * out_channels) #Overidding skip connection to identity #not needed as root_weight=False
        self.lin_value = nn.Identity(in_channels, heads * out_channels)  # Overidding channel mixing to identity
        self.stored_ei = None
        self.stored_alpha = None

    def forward(self, x, edge_index):
        AX, (self.stored_ei, self.stored_alpha) = super(GRAND_conv, self).forward(x, edge_index, return_attention_weights=True)
        return AX - x


# simple grand conv implementation from transformer conv
class GAT_plus(GATConv):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html#torch_geometric.nn.conv.TransformerConv
    def __init__(self, opt, in_channels, out_channels, heads=1, concat=False, beta=False, dropout=0,
                 edge_dim=None, bias=False, root_weight=False):
        super(GAT_plus, self).__init__(in_channels, out_channels, heads=1, concat=False, dropout=0.0,
                                         edge_dim=None, bias=False, root_weight=False) #nb these are non default values

        self.opt = opt
        self.lin_src = nn.Identity(in_channels, heads * out_channels)  # Overidding channel mixing to identity
        self.lin_dst = nn.Identity(in_channels, heads * out_channels)
        self.stored_ei = None
        self.stored_alpha = None

    def forward(self, x, edge_index):
        if self.opt['gat_plus_type'] == 'GAT_res_lap':
            GAT_x1, (self.stored_ei, self.stored_alpha) = super(GAT_plus, self).forward(x, edge_index, return_attention_weights=True)
            # calc AX with stored_alpha and x using spmm
            sparse_alpha = torch.sparse_coo_tensor(self.stored_ei, self.stored_alpha.squeeze(1), (x.shape[0], x.shape[0]))
            # row_sum = torch.sparse.sum(sparse_alpha, dim=0).to_dense() #check need transpose
            Ax = torch.sparse.mm(sparse_alpha.T, x)

            return Ax - x


        elif self.opt['gat_plus_type'] == 'GAT_lin':
            GAT_x1, (self.stored_ei, self.stored_alpha) = super(GAT_plus, self).forward(x, edge_index, return_attention_weights=True)
            sparse_alpha = torch.sparse_coo_tensor(self.stored_ei, self.stored_alpha.squeeze(1), (x.shape[0], x.shape[0]))
            # row_sum = torch.sparse.sum(sparse_alpha, dim=0).to_dense() #check need transpose
            Ax = torch.sparse.mm(sparse_alpha.T, x)

            return Ax



