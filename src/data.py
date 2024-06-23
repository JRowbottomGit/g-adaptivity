import os
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from firedrake.pyplot import plot, tripcolor
from utils_eval import eval_grid_MMPDE_MA

import sys
sys.path.append('../')

from firedrake_difFEM.solve_poisson import poisson2d_fmultigauss_bcs, poisson1d_fmultigauss_bcs
from firedrake_difFEM.difFEM_poisson_1d import torch_FEM_1D, u_true_exact_1d
from firedrake import UnitIntervalMesh, UnitSquareMesh, FunctionSpace, DirichletBC, CheckpointFile

from classical_meshing.ma_mesh_2d import MA2d, MMPDE5_2d
from classical_meshing.ma_mesh_1d import MMPDE5_1d

from params_poisson import get_params, run_params
from utils_data import make_data_name, to_float32, convert_to_boundary_mask, map_firedrake_to_cannonical_ordering_2d, map_firedrake_to_cannonical_ordering_1d
from utils_eval import update_mesh_coords


class PyG_Dataset(object):
  def __init__(self, data):
    self.data = data
    self.num_nodes = data.x_comp.shape[0]
    self.num_node_features = data.x_comp.shape[1]

class MeshInMemoryDataset(pyg.data.InMemoryDataset):
    def __init__(self, root, train_test, num_data, mesh_dims, opt, transform=None, pre_transform=None):
        self.root = root
        self.train_test = train_test
        self.num_data = num_data

        self.opt = opt
        self.dim = len(mesh_dims)
        if self.dim == 1:
            self.n = mesh_dims[0]
        elif self.dim == 2:
            self.n = mesh_dims[0]
            self.m = mesh_dims[1]
        self.num_x_comp_features = self.dim
        self.num_x_phys_features = self.dim
        self.mesh = None
        self.x_comp_shared = None
        self.mapping_dict = None
        self.mapping_tensor = None
        self.mapping_dict_fine = None
        self.mapping_tensor_fine = None

        super(MeshInMemoryDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        custom_attributes_path = os.path.join(self.root, "processed", "custom_attributes.pt")
        if os.path.exists(custom_attributes_path):
            custom_attributes = torch.load(custom_attributes_path)
            self.x_comp_shared = custom_attributes['x_comp_shared']
            self.mapping_tensor = custom_attributes['mapping_tensor']
            self.mapping_dict = custom_attributes['mapping_dict']
            self.mapping_tensor_fine = custom_attributes['mapping_tensor_fine']
            self.mapping_dict_fine = custom_attributes['mapping_dict_fine']
            self.orig_opt = custom_attributes['orig_opt']

        # Load the mesh
        mesh_file_path = os.path.join(self.root, "processed", "mesh.h5")
        if os.path.exists(mesh_file_path):
            with CheckpointFile(mesh_file_path, 'r') as mesh_file:
                self.mesh = mesh_file.load_mesh("ref_mesh")

        deformed_mesh_file_path = os.path.join(self.root, "processed", "deformed_mesh.h5")
        if os.path.exists(deformed_mesh_file_path):
            with CheckpointFile(deformed_mesh_file_path, 'r') as mesh_file:
                self.mesh_deformed = mesh_file.load_mesh("deformed_mesh")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        opt = self.opt
        if self.dim == 1:
            n = self.n
            self.mesh = UnitIntervalMesh(n - 1, name="ref_mesh")
            self.mesh_deformed = UnitIntervalMesh(n - 1, name="deformed_mesh")
            self.fine_mesh = UnitIntervalMesh(opt['eval_quad_points'] - 1, name="fine_mesh")

        elif self.dim == 2:
            n = self.n
            m = self.m
            self.mesh = UnitSquareMesh(n - 1, m - 1, name="ref_mesh")
            self.mesh_deformed = UnitSquareMesh(n - 1, m - 1, name="deformed_mesh")
            self.fine_mesh = UnitSquareMesh(opt['eval_quad_points'] - 1, opt['eval_quad_points'] - 1, name="fine_mesh")

        with CheckpointFile(os.path.join(self.root, "processed", f"mesh.h5"), 'w') as mesh_file:
            mesh_file.save_mesh(self.mesh)
        with CheckpointFile(os.path.join(self.root, "processed", f"deformed_mesh.h5"), 'w') as mesh_file:
            mesh_file.save_mesh(self.mesh_deformed)

        self.x_comp_shared = torch.tensor(self.mesh.coordinates.dat.data_ro)
        self.x_fine_shared = torch.tensor(self.fine_mesh.coordinates.dat.data_ro)

        if self.dim == 1:
            mapping_dict, mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(self.x_comp_shared, n)
            mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, X_fd_vec_fine = map_firedrake_to_cannonical_ordering_1d(self.x_fine_shared, self.opt['eval_quad_points'])
        elif self.dim == 2:
            mapping_dict, mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(self.x_comp_shared, n, m)
            mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, Y_fd_grid_fine, X_fd_vec_fine, Y_fd_vec_fine = map_firedrake_to_cannonical_ordering_2d(self.x_fine_shared, self.opt['eval_quad_points'], self.opt['eval_quad_points'])

        self.mapping_dict = mapping_dict
        self.mapping_tensor = mapping_tensor
        self.mapping_dict_fine = mapping_dict_fine
        self.mapping_tensor_fine = mapping_tensor_fine

        custom_attributes = {
            'x_comp_shared': self.x_comp_shared,
            'mapping_dict': self.mapping_dict,
            'mapping_tensor': self.mapping_tensor,
            'mapping_dict_fine': self.mapping_dict_fine,
            'mapping_tensor_fine': self.mapping_tensor_fine,
            'orig_opt': opt,
        }
        torch.save(custom_attributes, os.path.join(self.root, "processed", "custom_attributes.pt"))

        idx = 0
        data_list = []
        if opt['data_type'] in ['fixed']:
            num_data_dict = {1: 9, 2: 25}
            self.num_data = num_data_dict[self.dim]

        for i in range(self.num_data):
            successful_eval = True
            data = firedrake_mesh_to_PyG(self.mesh)

            c_list = []
            s_list = []
            if opt['data_type'] in ['randg']:
                for j in range(opt['num_gauss']):
                        if opt['data_burgers']:
                            s = np.random.uniform(opt['scale']*0.5, opt['scale']*2.0, self.dim).astype('f')
                            c = np.random.uniform(opt['scale']*opt['burgers_limits'], 1-opt['scale']*opt['burgers_limits'], self.dim).astype('f')  # float to match torch precison
                            c_list.append(c)
                            s_list.append(s)
                        else:
                            c = np.random.uniform(0, 1, self.dim).astype('f') #float to match torch precison
                            s = np.random.uniform(0.1, 0.5, self.dim).astype('f')
                            c_list.append(c)
                            s_list.append(s)

            elif opt['data_type'] == 'fixed':
                if self.dim == 1: #9 interrior points in 0.1-0.9 grid and iterate over them
                    x_coord = (i + 1) / (num_data_dict[self.dim] + 1)
                    c1 = np.array([x_coord]).astype('f') #float to match torch precison
                    s1 = np.array([opt['scale']]).astype('f')
                    c_list.append(c1)
                    s_list.append(s1)
                    if opt['num_gauss'] == 2:
                        c2 = np.array([0.5])
                        s2 = np.array([opt['scale']])
                        c_list.append(c2)
                        s_list.append(s2)
                elif self.dim == 2: #25 interrior points in 0.1-0.9 grid and iterate over them plus a fixed central Gaussian
                    x_coord1 = i % 5 * 0.2 + 0.1
                    y_coord1 = i // 5 * 0.2 + 0.1
                    c1 = np.array([x_coord1, y_coord1])
                    s1 = np.array([opt['scale'], opt['scale']])
                    c_list.append(c1)
                    s_list.append(s1)
                    if opt['num_gauss'] == 2:
                        c2 = np.array([0.5, 0.5])
                        s2 = np.array([opt['scale'], opt['scale']])
                        c_list.append(c2)
                        s_list.append(s2)

            pde_params = {'centers': c_list, 'scales': s_list}
            pde_params['scale_list'] = s_list
            if opt['data_type'] not in ['randg']:
                if self.dim == 1:
                    pde_params['scale_value'] = s_list[0]  # just for naming
                elif self.dim == 2:
                    pde_params['scale_value'] = s_list[0][0]  # just for naming
            pde_params['mon_power'] = opt['mon_power']
            pde_params['mesh_type'] = opt['mesh_type']
            pde_params['mon_reg'] = opt['mon_reg']
            pde_params['num_gauss'] = opt['num_gauss']
            pde_params['eval_quad_points'] = opt['eval_quad_points']
            pde_params['fast_M2N_monitor'] = self.opt['fast_M2N_monitor']
            if self.opt['M2N_alpha'] is not None:
                pde_params['M2N_alpha'] = self.opt['M2N_alpha']
            if self.opt['M2N_beta'] is not None:
                pde_params['M2N_beta'] = self.opt['M2N_beta']
            data.pde_params = pde_params

            #deform mesh using MMPDE/MA
            if self.dim == 1:
                data.x_phys, data.ma_its, data.build_time = deform_mesh_mmpde1d(self.x_comp_shared, n, pde_params)
            elif self.dim == 2:
                if opt['dataset'] in ['fd_ma_2d','fd_M2N_2d']:
                    x_phys, data.ma_its, data.build_time = deform_mesh_ma2d(self.x_comp_shared, n, m, pde_params)
                    data.x_phys = torch.from_numpy(x_phys)
                elif opt['dataset'] == 'fd_mmpde_2d':
                    data.x_phys, data.ma_its, data.build_time = deform_mesh_mmpde2d(self.x_comp_shared, n, m, pde_params)

            num_meshpoints = n if self.dim == 1 else n

            if self.dim == 1:
                eval_fct = poisson1d_fmultigauss_bcs
                fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, self.mesh, self.mesh_deformed, self.fine_mesh, eval_fct, self.dim, num_meshpoints, c_list, s_list, opt)
            elif self.dim == 2:
                eval_fct = poisson2d_fmultigauss_bcs
                x_values = np.linspace(0, 1, opt['eval_quad_points'])
                y_values = np.linspace(0, 1, opt['eval_quad_points'])
                X, Y = np.meshgrid(x_values, y_values)
                eval_vec = np.reshape(np.array([X, Y]), [2, opt['eval_quad_points'] ** 2])

                fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, self.mesh, self.mesh_deformed, self.fine_mesh, eval_fct, self.dim, num_meshpoints, c_list, s_list, opt, eval_vec, X, Y)

                if fcts_on_grids_dict['uu_ma'] == 0.:
                    successful_eval = False
                    print("Error in eval_grid_MMPDE_MA, saving None's")

            #fine eval (saving tensors for fast pde loss)
            if self.dim == 1:
                uu_fine, u_true_fine, f_fine = poisson1d_fmultigauss_bcs(self.fine_mesh, c_list, s_list)
            elif self.dim == 2:
                uu_fine, u_true_fine, f_fine = poisson2d_fmultigauss_bcs(self.fine_mesh, c_list, s_list, rand_gaussians=False)

            data.eval_errors = eval_errors_dict
            data.successful_eval = successful_eval

            # save the firedrake functions to file
            if opt['data_type'] in ['randg']:
                filename_suffix = f"dim_{self.dim}_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data.h5"
            else:
                filename_suffix = f"dim_{self.dim}_scale_{round(data.pde_params['scale_value'], 2)}" \
                                  f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data.h5"

            pde_data_file = os.path.join(self.root, "processed", filename_suffix)

            with CheckpointFile(pde_data_file, 'w') as pde_file:
                pde_file.save_mesh(self.mesh)
                pde_file.save_mesh(self.mesh_deformed)
                pde_file.save_function(fcts_on_grids_dict['uu_grid'], name="uu")
                pde_file.save_function(fcts_on_grids_dict['u_true_grid'], name="u_true")
                pde_file.save_function(fcts_on_grids_dict['f_grid'], name="f")
                if data.successful_eval:
                    pde_file.save_function(fcts_on_grids_dict['uu_ma'], name="uu_ma")
                    pde_file.save_function(fcts_on_grids_dict['u_true_ma'], name="u_true_ma")
                    pde_file.save_function(fcts_on_grids_dict['f_ma'], name="f_ma")

            #also saving the torch tensors
            #note we delay mapping firedrake functions to cannonical ordering to maintain consistency with the mesh and x_comp/phys
            data.uu_tensor = torch.from_numpy(fcts_on_grids_dict['uu_grid'].dat.data)
            data.u_true_tensor = torch.from_numpy(fcts_on_grids_dict['u_true_grid'].dat.data)
            data.f_tensor = torch.from_numpy(fcts_on_grids_dict['f_grid'].dat.data)
            data.uu_fine_tensor = torch.from_numpy(uu_fine.dat.data)
            data.u_true_fine_tensor = torch.from_numpy(u_true_fine.dat.data)
            data.f_fine_tensor = torch.from_numpy(f_fine.dat.data)
            if data.successful_eval:
                data.uu_MA_tensor = torch.from_numpy(fcts_on_grids_dict['uu_ma'].dat.data)
                data.u_true_MA_tensor = torch.from_numpy(fcts_on_grids_dict['u_true_ma'].dat.data)
                data.f_MA_tensor = torch.from_numpy(fcts_on_grids_dict['f_ma'].dat.data)
            else:
                data.uu_MA_tensor = torch.tensor([0.])
                data.u_true_MA_tensor = torch.tensor([0.])
                data.f_MA_tensor = torch.tensor([0.])

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            idx += 1

        data, slices = self.collate(data_list)
        data.apply(to_float32)
        torch.save((data, slices), self.processed_paths[0])


    def get(self, idx):
        data = super().get(idx)
        data.x_comp = self.x_comp_shared.float()
        if isinstance(data.x_phys, np.ndarray):
            data.x_phys = torch.from_numpy(data.x_phys)

        mon_power = round(data.pde_params['mon_power'].item(), 2)
        num_gauss = data.pde_params['num_gauss'].item()
        if 'mon_reg' in data.pde_params:
            mon_reg = round(data.pde_params['mon_reg'].item(), 2)
        if self.opt['data_type'] in ['randg']:
            filename_suffix = f"dim_{self.dim}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data.h5"
        else:
            # Retrieve scale and mon_power from data object
            scale = round(data.pde_params['scale_value'].item(), 2)
            filename_suffix = f"dim_{self.dim}_scale_{scale}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data.h5"
        pde_data_file = os.path.join(self.root, "processed", filename_suffix)

        with CheckpointFile(pde_data_file, 'r') as pde_file:
            self.mesh = pde_file.load_mesh("ref_mesh")
            self.mesh_deformed = pde_file.load_mesh("deformed_mesh")
            uu = pde_file.load_function(self.mesh, "uu")
            u_true = pde_file.load_function(self.mesh, "u_true")
            f = pde_file.load_function(self.mesh, "f")
            if data.successful_eval:
                uu_ma = pde_file.load_function(self.mesh_deformed, "uu_ma")
                u_true_ma = pde_file.load_function(self.mesh_deformed, "u_true_ma")
                f_ma = pde_file.load_function(self.mesh_deformed, "f_ma")
            else:
                uu_ma, u_true_ma, f_ma = 0., 0., 0.

        data.uu = uu
        data.u_true = u_true
        data.f = f
        if data.successful_eval:
            data.uu_ma = uu_ma
            data.u_true_ma = u_true_ma
            data.f_ma = f_ma
        else:
            data.uu_ma = 0.
            data.u_true_ma = 0.
            data.f_ma = 0.

        return data

#generate a regular n x m mesh as a PyG data - not used anymore instead load from Firedrake
def generate_mesh_2d(n, m):
    ''' n for x axis, m for y axis '''
    x_points = torch.linspace(0, 1, n + 1)
    y_points = torch.linspace(0, 1, m + 1)
    x, y = torch.meshgrid(x_points, y_points)
    x = x.reshape(-1)
    y = y.reshape(-1)
    points = torch.stack([x, y], dim=1)
    edges = []

    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and j < m:
                edges.append([i * (m + 1) + j, i * (m + 1) + j + 1])
                edges.append([i * (m + 1) + j, (i + 1) * (m + 1) + j])
            elif i < n:
                edges.append([i * (m + 1) + j, (i + 1) * (m + 1) + j])
            elif j < m:
                edges.append([i * (m + 1) + j, i * (m + 1) + j + 1])
    edges = torch.tensor(edges).T

    #make edges undirected
    edges = torch.cat([edges, torch.flip(edges, dims=[0])], dim=1)

    boundary_nodes = set()
    for i in range(n + 1):
        boundary_nodes.add(i) #bottom row
        boundary_nodes.add(m * (n + 1) + i) #top row
    for i in range(m + 1):
        boundary_nodes.add(i * (m + 1))  # left column
        boundary_nodes.add(n + i * (n + 1)) # right column

    boundary_nodes = list(boundary_nodes)

    boundary_nodes = convert_to_boundary_mask(boundary_nodes, num_nodes=(n + 1) * (m + 1))

    return PyG_Dataset(pyg.data.Data(x_comp=points, x_phys=points,
                                          edge_index=edges, n=n, m=m, num_node_features=2, boundary_nodes=boundary_nodes))


#deform the mesh for random displacement bounded by 1 / n
def deform_mesh_perturb(dataset, opt):
    #adding noise to x_comp and creating un-noised x_phys
    n = dataset.data.n
    m = dataset.data.m
    #torch sample uniform noise for each node between 0 and half the distance between nodes
    eps = torch.rand(dataset.data.x_comp.shape) / (2 * n) * 2
    dataset.data.x_phys = dataset.data.x_comp.clone()

    if opt['fix_boundary']:
        #index for not boundary nodes
        not_boundary_nodes = [i for i in range((n + 1) * (m + 1)) if i not in dataset.data.boundary_nodes]
        dataset.data.x_comp[not_boundary_nodes] = dataset.data.x_comp[not_boundary_nodes] + eps[not_boundary_nodes]
    else:
        dataset.data.x_comp = dataset.data.x_comp + eps

    return dataset


def deform_mesh_mmpde1d(x_comp, n, opt):
    # deform the mesh with MMPDE5 in 1d
    mapping_dict, mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(x_comp, n)
    coords_mmpde5 = [None]
    coords_mmpde5[0], j, build_time = MMPDE5_1d(X_fd_grid, n, opt)
    x_phys = coords_mmpde5[0]

    return x_phys, j, build_time


def deform_mesh_mmpde2d(x_comp, n, m, pde_params):
    mapping_dict, mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(x_comp, n, m)
    coords_mmpde5 = [None, None]
    coords_mmpde5[0], coords_mmpde5[1], j, build_time = MMPDE5_2d(X_fd_grid, Y_fd_grid, n, pde_params)
    x_phys = torch.zeros_like(x_comp)

    for i in range(n):
        for j in range(m):
            x_phys[mapping_dict[(i, j)],0] = coords_mmpde5[0][(i, j)]
            x_phys[mapping_dict[(i, j)],1] = coords_mmpde5[1][(i, j)]

    j = j + 1 #to account for the initial mesh
    return x_phys, j, build_time


def deform_mesh_ma2d(x_comp, n, m, pde_params):
    x_phys, j, build_time = MA2d(x_comp, n, pde_params)
    j=j+1 #to account for the initial mesh
    return x_phys, j, build_time

def firedrake_mesh_to_PyG(mesh):
    # Get coordinates of the vertices
    coordinates = mesh.coordinates.dat.data_ro
    # Get the cell to node mapping
    cell_node_map = mesh.coordinates.cell_node_map().values
    # Initialize a set for edges (each edge represented as a tuple)
    edges_set = set()
    # Iterate through each cell
    for cell in cell_node_map:
        # For each pair of nodes in the cell, add an edge
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                # Add edge in both directions to ensure it's undirected
                edges_set.add((cell[i], cell[j]))
                edges_set.add((cell[j], cell[i]))

    # Convert edge set to a tensor
    edge_index = torch.tensor(list(edges_set), dtype=torch.long).t().contiguous()
    # Define a function space on the mesh
    V = FunctionSpace(mesh, "CG", 1)
    # Create a boundary condition
    bc = DirichletBC(V, 0, "on_boundary")
    # Get the boundary nodes
    boundary_nodes = bc.nodes

    boundary_nodes_mask = convert_to_boundary_mask(boundary_nodes, num_nodes=len(coordinates))

    boundary_nodes_dict = {}
    unique_boundary_ids = mesh.topology.exterior_facets.unique_markers
    for boundary_id in unique_boundary_ids:
        bc = DirichletBC(V, 0, boundary_id)
        boundary_nodes_dict[boundary_id] = bc.nodes

    boundary_ids = list(unique_boundary_ids)
    all_boundary_nodes = []
    for bid in boundary_ids:
        all_boundary_nodes.extend(V.boundary_nodes(bid))
    unique_nodes, counts = np.unique(all_boundary_nodes, return_counts=True)
    corner_nodes = unique_nodes[counts > 1]

    #mask for edges who's dst node is in the boundary and source node is in the interior
    to_boundary_edge_mask = torch.tensor([edge_index[1][i].item() in boundary_nodes and edge_index[0][i].item() not in boundary_nodes for i in range(edge_index.shape[1])])
    #mask for edges who's dst node is in the corner
    # to_corner_nodes_mask = torch.tensor([edge_index[1][i].item() in corner_nodes and edge_index[0][i].item() not in boundary_nodes for i in range(edge_index.shape[1])])
    to_corner_nodes_mask = torch.tensor([edge_index[1][i].item() in corner_nodes for i in range(edge_index.shape[1])])

    # Invert the boundary_nodes_dict to get node:boundary_id mapping
    node_boundary_map = {}
    for boundary_id, nodes in boundary_nodes_dict.items():
        for node in nodes:
            if node not in node_boundary_map.keys():
                node_boundary_map[node] = [boundary_id]
            else:
                node_boundary_map[node].append(boundary_id)

    # Create a mask for edges between different boundaries, excluding corner nodes
    diff_boundary_edges_mask_list = []
    for edge in edge_index.t().tolist():
        src_node, dst_node = edge
        if src_node in node_boundary_map.keys() and dst_node in node_boundary_map.keys():
            src_boundary_list = node_boundary_map[src_node]
            dst_boundary_list = node_boundary_map[dst_node]
            # Check if both nodes in edge are on different boundaries and neither is a corner node
            if src_boundary_list != dst_boundary_list and src_node not in corner_nodes and dst_node not in corner_nodes:
                diff_boundary_edges_mask_list.append(True)
            else:
                diff_boundary_edges_mask_list.append(False)
        else:
            diff_boundary_edges_mask_list.append(False)

    diff_boundary_edges_mask = torch.tensor(diff_boundary_edges_mask_list)

    # Create the PyG graph
    data = pyg.data.Data(x_comp=torch.tensor(coordinates, dtype=torch.float),
                            x_phys=torch.tensor(coordinates, dtype=torch.float), edge_index=edge_index,
                            boundary_nodes=boundary_nodes_mask, corner_nodes=corner_nodes, to_boundary_edge_mask=to_boundary_edge_mask,
                            to_corner_nodes_mask=to_corner_nodes_mask, diff_boundary_edges_mask=diff_boundary_edges_mask,
                            boundary_nodes_dict=boundary_nodes_dict, node_boundary_map=node_boundary_map)
    return data


def plot_initial_dataset_2d(dataset, opt, plot_mesh=True, plot_fem0=True, plot_fem1p=False):
    title_suffix = f"{opt['data_name']}"
    # Create a DataLoader with batch size of 1 to load one data point at a time
    loader = DataLoader(dataset, batch_size=1, shuffle=False) #True)

    if plot_mesh:
        # figure for mesh
        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
        # fig.suptitle(f'Mesh {title_suffix}', fontsize=20)
        fig.suptitle(f"Monge Ampere mesh - {opt['mesh_dims']} regularisation {opt['mon_reg']}", fontsize=20)
        axs = axs.ravel()

    if plot_fem0:
        #figure for firedrake function
        fig2, axs2 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
        # fig2.suptitle(f'uu on reg mesh {title_suffix}', fontsize=20)
        fig2.suptitle(f"FEM solution for Gaussian scales {opt['scale']}", fontsize=20)
        #add small white space between title and plot
        fig2.subplots_adjust(top=0.2)
        axs2 = axs2.ravel()

    if plot_fem1p:
        #figure for torch tensor
        fig3, axs3 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
        fig3.suptitle(f'FEM uu on mesh {title_suffix}', fontsize=20)
        axs3 = axs3.ravel()

    # Loop over the dataset
    for i, data in enumerate(loader):
        if i == 25:
            break
        if plot_mesh:
            # Convert PyG graph to NetworkX graph
            G = to_networkx(data, to_undirected=True)
            # Get node positions from the coordinates attribute in the PyG graph
            x = data.x_phys
            positions = {i: x[i].tolist() for i in range(x.shape[0])}
            # Draw the graph with reduced tick size and no labels
            nx.draw(G, pos=positions, ax=axs[i], node_size=1, width=0.5, with_labels=False)

        if plot_fem0:
            #plot the FireDrake function
            colors = tripcolor(data.uu[0], axes=axs2[i])#, shading='gouraud', cmap='viridis')

        if plot_fem1p:
            #plot the torch tensor
            x_comp_cannon = data.x_comp[dataset.mapping_tensor].cpu().numpy()
            uu_cannon = data.uu_tensor[dataset.mapping_tensor].cpu().numpy()
            # scatter = axs2[i].scatter(x_comp[:, 0], x_comp[:, 1], c=uu, cmap='viridis')
            # plt.colorbar(scatter, ax=axs2[i])
            contourf = axs3[i].tricontourf(x_comp_cannon[:, 0], x_comp_cannon[:, 1], uu_cannon, levels=15, cmap='viridis')
            # plt.colorbar(contourf, ax=axs3[i])

    #saves figs
    if plot_mesh:
        fig.tight_layout()
        fig.savefig(f"../data/{opt['data_name']}_mesh.pdf", format='pdf', dpi=300, bbox_inches='tight')
    if plot_fem0:
        fig2.tight_layout()
        fig2.savefig(f"../data/{opt['data_name']}_uu.pdf", format='pdf', dpi=300, bbox_inches='tight')
    if plot_fem1p:
        fig3.tight_layout()
        fig3.savefig(f"../data/{opt['data_name']}_uu_torch.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


def plot_initial_dataset_1d(dataset, opt):
    mesh = dataset.mesh
    dim = len(dataset.opt['mesh_dims'])
    num_meshpoints = dataset.opt['mesh_dims'][0] if dim == 1 else dataset.opt['mesh_dims'][0] * dataset.opt['mesh_dims'][1]
    fine_mesh = UnitIntervalMesh(opt['eval_quad_points'] - 1, name="fine_mesh")

    # Create a DataLoader with batch size of 1 to load one data point at a time
    loader = DataLoader(dataset, batch_size=1)

    #figure for FEM on regular mesh
    fig0, axs0 = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))  # adjust as necessary
    axs0 = axs0.ravel()
    fig0.suptitle('FEM on regular mesh', fontsize=20)
    fig0.tight_layout()

    # #figure for FEM on MMPDE5 mesh
    fig1, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))  # adjust as necessary
    axs1 = axs.ravel()
    fig1.suptitle('MMPDE5 mesh', fontsize=20)
    fig1.tight_layout()

    # Loop over the dataset
    for i, data in enumerate(loader):
        if i == 9:
            break
        if opt['overfit_num']:
            if i not in opt['overfit_num']:
                continue  # skip to next batch
            else:
                print(f"Overfitting on batch {i} of {opt['overfit_num']}")

        data.idx = i
        #this happens for numpy arrays in PyG datasets
        c_list = data.pde_params['centers'][0]
        s_list = data.pde_params['scales'][0]
        c_list_torch = [torch.from_numpy(c_0) for c_0 in data.pde_params['centers'][0]]
        s_list_torch = [torch.from_numpy(s_0) for s_0 in data.pde_params['scales'][0]]

        #gen fine baseline true solution
        uu_fine, u_true_fine, _ = poisson1d_fmultigauss_bcs(fine_mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)

        # 1) plot the FEM on regular mesh
        plot(data.uu[0], axes=axs0[i], label='uu_fem_xcomp', color='orange')
        plot(data.u_true[0], axes=axs0[i], label='u_true_xcomp', color='green')
        plot(uu_fine, axes=axs0[i], label='uu_fem_fine', color='lightblue')
        plot(u_true_fine, axes=axs0[i], label='u_true_fine', color='grey')

        if opt['solver'] == 'firedrake':
            uuFD_coarse, u_true_FD_coarse, _, _ = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
            plot(uuFD_coarse, axes=axs0[i], label='FD_out', color='purple')

        elif opt['solver'] == 'torch_FEM':
            mesh_points = torch.from_numpy(mesh.coordinates.dat.data_ro).float()
            quad_points = torch.from_numpy(fine_mesh.coordinates.dat.data_ro).float()
            UUtorchSolcoeffs, mesh_points, uutorch_coarse2fine, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list_torch, s_list_torch)
            full_UUtorchsol = torch.cat((BC1, UUtorchSolcoeffs.squeeze(), BC2), 0)
            axs0[i].plot(mesh_points, full_UUtorchsol, label='torchFEM_out', color='purple')


        #scatter plot of data.u_true on the regular mesh
        u_true_x_comp = u_true_exact_1d(data.x_comp, c_list, s_list).to('cpu').detach().numpy()
        axs1[i].scatter(data.x_comp, u_true_x_comp, color='red', marker='x', label='u_true_x_comp')
        # axs0[i].scatter(data.x_comp, data.u_true[0].dat.data_ro, color='red', marker='x', label='u_true_x_comp')

        #extra ticks for the x axis to show the mesh points
        extraticks = data.x_comp.tolist()
        dash_length = 0.04
        ymin = -0.02
        dashcol = 'black'
        dashwid = 2.
        for tick in extraticks:
            axs0[i].plot([tick, tick], [ymin, ymin + dash_length], color=dashcol, linestyle='-', linewidth=dashwid)
        axs0[i].legend()

        # 2) plot the FEM on MMPDE5/MA (target phys) mesh
        mesh = dataset.mesh
        x = data.x_phys
        update_mesh_coords(mesh, x)

        uu_ma, u_true_ma, _ = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)

        if opt['solver'] == 'firedrake':
            uu_MA_coarse, u_true_MA_coarse, _, _ = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
            plot(uu_MA_coarse, axes=axs1[i], label='FD_out', color='purple')

        elif opt['solver'] == 'torch_FEM':
            mesh_points = torch.from_numpy(mesh.coordinates.dat.data_ro).float()
            quad_points = torch.from_numpy(fine_mesh.coordinates.dat.data_ro).float()
            MMPDESolcoeffs, MMPDEmesh_points, uu_MA_coarse2fine, _, _ = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list_torch, s_list_torch)
            full_MMPDEUUsol = torch.cat((BC1, MMPDESolcoeffs.squeeze(), BC2), 0)
            axs1[i].plot(mesh_points, full_MMPDEUUsol, label='torchFEM_out', color='purple')

        # colors = tripcolor(uu_ma, axes=axs2[i])#, shading='gouraud', cmap='viridis')
        plot(uu_ma, axes=axs1[i], label='uu_fem_ma', color='orange')
        plot(u_true_ma, axes=axs1[i], label='u_true_ma', color='green')
        plot(uu_fine, axes=axs1[i], label='uu_fem_fine', color='lightblue')
        plot(u_true_fine, axes=axs1[i], label='u_true_fine', color='grey')

        u_true_x_comp = u_true_exact_1d(data.x_comp, c_list, s_list).to('cpu').detach().numpy()
        u_true_x_phys = u_true_exact_1d(data.x_phys, c_list, s_list).to('cpu').detach().numpy()
        axs1[i].scatter(data.x_comp, u_true_x_comp, color='red', marker='x', label='u_true_x_comp')
        axs1[i].scatter(data.x_phys, u_true_x_phys, color='blue', marker='x', label='u_true_MA')

                        #x-axis dashed for the x_phys
        extraticks = data.x_phys.tolist()
        # axs1[i].set_xticks(list(axs1[i].get_xticks()) + extraticks)
        for tick in extraticks:
            axs1[i].plot([tick, tick], [ymin, ymin + dash_length], color=dashcol, linestyle='-', linewidth=dashwid)
        axs1[i].legend()
    #saves figs
    fig0.tight_layout()
    fig0.savefig(f"../data/{opt['data_name']}_uu.pdf", format='pdf', dpi=300, bbox_inches='tight')
    fig1.tight_layout()
    fig1.savefig(f"../data/{opt['data_name']}_uu_ma.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # mesh = UnitSquareMesh(10, 10) #note this is 10 squares in each direction, so 11 nodes in each direction
    # data = firedrake_mesh_to_PyG(mesh)
    # plot2d(data)    # Plot the graph
    # triplot(mesh)    # Plot the mesh

    # # Add labels for nodes
    # coords = mesh.coordinates.dat.data
    # for i, (x, y) in enumerate(coords):
    #     plt.text(x, y, str(i), color='red')  # Adjust the color, position, etc. as needed
    # plt.show()

    # # 3D cube box
    # mesh = BoxMesh(10, 10, 10, 1, 1, 1)
    # data = firedrake_mesh_to_PyG(mesh)
    # plot_3d_pyg_graph_interactive(data)

    opt = get_params()
    # opt = tf_ablation_args(opt)
    if not opt['wandb_sweep']:
        opt = run_params(opt)

    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed

    data_dim = 2 #1 #2
    if data_dim == 1:     # #1D
        opt['data_type'] = "randg"# "all" #'randg'
        opt['mesh_type'] = "mmpde" #
        opt['dataset'] = f"fd_{opt['mesh_type']}_1d"
        opt['mesh_dims'] = [11]#10] #6]#11]21]51]101]#in terms of total number of nodes
        opt['mon_reg'] = 1.#1. #0.10#0.01
        opt['num_gauss'] = 1
        opt['num_train'] = 100
        opt['num_test'] = 100

    elif data_dim == 2:        # 2D
        opt['data_type'] = "randg"#"randg" #"randg"# "all"#"randg"# "all" #'randg'
        opt['mesh_type'] = "ma"  # ma or mmpde or M2N
        if opt['mesh_type'] == "M2N":
            opt['fast_M2N_monitor'] = "fast" #"slow" #"superslow" "fast"
            opt['M2N_alpha'] = 1.0
            opt['M2N_beta'] = 1.0
        opt['dataset'] = f"fd_{opt['mesh_type']}_2d" #'fd_ma_grid_2d' #'fd_ma_L'#'fd_noisey_grid' #fd_ma_grid#'noisey_grid'
        opt['mesh_dims'] = [11, 11] #[15, 15] #[11, 11]
        opt['mon_reg'] = 0.1 #.1 #0.1 #0.01


    for train_test in ['train', 'test']:
        opt = make_data_name(opt, train_test)
        if train_test == 'train':
            dataset = MeshInMemoryDataset(f"../data/{opt['data_name']}", train_test, opt['num_train'], opt['mesh_dims'], opt)
        elif train_test == 'test':
            dataset = MeshInMemoryDataset(f"../data/{opt['data_name']}", train_test, opt['num_test'], opt['mesh_dims'], opt)

        plot_initial_dataset_2d(dataset, opt)
