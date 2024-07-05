import os
import numpy as np
import random
import torch
import wandb
import matplotlib.pyplot as plt

from params import get_params, run_params, get_arg_list

def to_float32(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float64:
            return obj.float()
        return obj
    elif isinstance(obj, dict):
        return {key: to_float32(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_float32(item) for item in obj]
    # Handle other types as needed
    return obj


def convert_to_boundary_mask(boundary_nodes, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[boundary_nodes] = 1
    return mask


def map_firedrake_to_cannonical_ordering_1d(x_comp, n):
    X_FD = x_comp
    # Rescale X and Y to [0, 1] if not already
    X_scaled = (X_FD - torch.min(X_FD)) / (torch.max(X_FD) - torch.min(X_FD))

    # Create cannonical-style grid
    X_grid = torch.linspace(0, 1, n)
    X_vec = X_grid.reshape(-1)

    # Map between firedrake ordering and regualr grid ordering
    mapping_dict = {}
    mapping_tensor = torch.zeros(x_comp.shape[0], dtype=torch.long)
    #loop over cannonical grid and find closest point in FD grid
    for i in range(n):
        cannon_idx = i
        x = X_grid[i]
        fd_index = torch.argmin((X_scaled - x) ** 2)
        mapping_tensor[cannon_idx] = fd_index
        mapping_dict[(i)] = fd_index.item()

    return mapping_dict, mapping_tensor, X_grid, X_vec


#todo this breaks for non-square data because x-y / i-j convention is not consistent
def map_firedrake_to_cannonical_ordering_2d(x_comp, n, m):
    X_FD, Y_FD = x_comp.T
    # Rescale X and Y to [0, 1] if not already
    X_scaled = (X_FD - torch.min(X_FD)) / (torch.max(X_FD) - torch.min(X_FD))
    Y_scaled = (Y_FD - torch.min(Y_FD)) / (torch.max(Y_FD) - torch.min(Y_FD))

    # Create cannonical-style grid
    X_grid, Y_grid = torch.meshgrid(torch.linspace(0, 1, m), torch.linspace(0, 1, n))
    #fine grid like np.mgrid[0:(2 * N -1), 0:(2 * N -1)] / (2 * N-2)
    X_vec = X_grid.reshape(-1)
    Y_vec = Y_grid.reshape(-1)

    # Map between firedrake ordering and regualr grid ordering
    mapping_dict = {}
    mapping_tensor = torch.zeros(x_comp.shape[0], dtype=torch.long)
    #loop over cannonical grid and find closest point in FD grid
    for i in range(n):
        for j in range(m):
            cannon_idx = i * m + j
            x, y = X_grid[i, j], Y_grid[i, j]
            fd_index = torch.argmin((X_scaled - x) ** 2 + (Y_scaled - y) ** 2)
            mapping_tensor[cannon_idx] = fd_index
            mapping_dict[(i, j)] = fd_index.item()

    return mapping_dict, mapping_tensor, X_grid, Y_grid, X_vec, Y_vec

def alt_diag_edges(mapping_dict, n, m, boundary_nodes):
    """
    Generate the set of additional diagonal edges from bottom-left to top-right
    for each square in the grid, including edges in both directions,
    using the mapping from canonical grid positions to Firedrake mesh indices.

    Parameters:
    - mapping_dict: Dictionary mapping (i, j) grid positions to Firedrake mesh indices.
    - n: Number of rows in the canonical grid.
    - m: Number of columns in the canonical grid.

    Returns:
    - A set of tuples, each representing an edge between two nodes in the Firedrake mesh.
      Includes both directions for each edge.
    """
    diag_edges = set()
    boundary_edges = set()
    # Iterate over squares in the grid
    for i in range(n - 1): #row
        for j in range(m - 1): #col
            # Bottom-left corner of the current square in canonical grid terms
            # bottom_left = (i + 1, j)
            bottom_left_can_idx = (i, j)
            # Top-right corner of the current square in canonical grid terms
            # top_right = (i, j + 1)
            top_right_can_idx = (i + 1, j + 1)

            # Translate these positions into Firedrake mesh indices using the mapping
            bottom_left_index = mapping_dict[bottom_left_can_idx]
            top_right_index = mapping_dict[top_right_can_idx]

            # Check if either node is a boundary node and the other is not (or both) then add to boundary_edges
            if (bottom_left_index in boundary_nodes and top_right_index not in boundary_nodes) or \
                    (top_right_index in boundary_nodes and bottom_left_index not in boundary_nodes) or \
                    (bottom_left_index in boundary_nodes and top_right_index in boundary_nodes):
                boundary_edges.add((bottom_left_index, top_right_index))
                boundary_edges.add((top_right_index, bottom_left_index))
            else:
                # Add the edge in both directions
                edge1 = (bottom_left_index, top_right_index)
                edge2 = (top_right_index, bottom_left_index)
                diag_edges.add(edge1)
                diag_edges.add(edge2)

    return diag_edges, boundary_edges

def reshape_fd_tensor_to_grid(u_true, mapping_tensor, mesh_dims, batch_size=1, dim=None):
    if dim == 1:
        u_true_grid_b = u_true.reshape(batch_size, -1) #[mapping_tensor].view(mesh_dims[0])
        u_true_grid = u_true_grid_b
        return u_true_grid

    elif dim == 2:
        # reshape in batches first - need to reshape for CNN ordering
        u_true_grid_b = u_true.reshape(batch_size, -1)
        indices_expanded = mapping_tensor.unsqueeze(0).expand(batch_size, -1)
        u_true_grid_b = torch.gather(u_true_grid_b, 1, indices_expanded)
        # then reshape 2nd dimension to the square
        u_true_grid2_b = u_true_grid_b.reshape(batch_size, mesh_dims[0], mesh_dims[1])
        # then flip and transpose
        u_true_grid3_b = torch.flip(torch.transpose(u_true_grid2_b, 1, 2), [1])
        # reshaping_test(data, x_comp, u_true, u_true_grid3_b, mapping_tensor, mesh_dims)
        return u_true_grid3_b

def reshape_grid_to_fd_tensor(u_true_grid3_b, mapping_tensor):
    # Reverse the transpose and flip operations
    # Assuming tensor shape is (batch, C, H, W)
    # Transpose H and W
    u_true_grid2_b = u_true_grid3_b.transpose(-1, -2)
    # Flip along H
    u_true_grid_b = torch.flip(u_true_grid2_b, [-2])
    # Reshape
    u_true_rolled = u_true_grid_b.reshape(u_true_grid_b.size(0), -1)

    # u_true_grid2_b = u_true_grid3_b.reshape(num_batches, -1)
    _, indices = torch.sort(mapping_tensor)

    # Reshape back to 1D tensor for each batch
    u_true_grid_b = u_true_rolled[:, indices]

    return u_true_grid_b


def reshaping_test(data, x_comp, u_true, u_true_grid3_b, mapping_tensor, mesh_dims):
    #mask to get the first batch of nodes
    batch_num = 0
    batch_mask = data.batch == batch_num
    x_comp = x_comp[batch_mask]
    u_true = u_true[batch_mask]

    u_true_grid = u_true[mapping_tensor].view(mesh_dims[0], mesh_dims[1])
    u_true_grid = torch.flip(torch.transpose(u_true_grid, 0, 1), [0])

    # figure for torch tensor
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))  # adjust as necessary
    contourf = axs.tricontourf(x_comp[:, 0], x_comp[:, 1], u_true, levels=15, cmap='viridis')
    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))  # adjust as necessary
    plt.imshow(u_true_grid, cmap='viridis', interpolation='nearest')
    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))  # adjust as necessary
    plt.imshow(u_true_grid3_b[0], cmap='viridis', interpolation='nearest')
    plt.show()


def interpolate_firedrake_tensor(u, x, mapping_tensor, batch_size, mesh_dims):

    #convert fd tensor to cannonical ordering
    uu_grid = reshape_fd_tensor_to_grid(u, mapping_tensor, mesh_dims, batch_size).unsqueeze(1)
    x_grid = reshape_fd_tensor_to_grid(x[:,0], mapping_tensor, mesh_dims, batch_size)
    y_grid = reshape_fd_tensor_to_grid(x[:,1], mapping_tensor, mesh_dims, batch_size)
    xy_grid = torch.stack((x_grid, y_grid), dim=3)
    xy_scaled_grid = xy_grid * 2 - 1

    #use torch grid sample to interpolate
    uu_interp_grid = torch.nn.functional.grid_sample(uu_grid, xy_scaled_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    #visual test
    #uu_grid[0][0].detach().numpy()
    #uu_interp_grid[0][0].detach().numpy()

    #convert back to fd ordering
    # uu_interp = uu_interp_grid[mapping_tensor].view(mesh_dims[0], mesh_dims[1])
    uu_interp = reshape_grid_to_fd_tensor(uu_interp_grid, mapping_tensor, batch_size)

    return uu_interp.view(-1)

def make_data_name(opt, train_test="test"):
    mesh_dims = get_arg_list(opt['mesh_dims'])
    data_dim = len(mesh_dims)
    #build data name
    decimal_places = 2 if opt['mon_reg'] < 0.1 else 1
    formatted_mon_reg = f"{opt['mon_reg']:.{decimal_places}f}"

    if opt['data_type'] == "all": #builds test set for all scales and powers
        #need to specify: data_type, mesh_type, mon_type, mesh_dims, mon_reg
        if data_dim == 1:
            data_name = f"{opt['data_type']}_1d_test_{mesh_dims[0]}_{formatted_mon_reg}reg"
        elif data_dim == 2:
            if opt['mesh_type'] == "M2N":
                alpha_beta = ''
                if opt['M2N_alpha'] is not None:
                    alpha_beta += f"_a{opt['M2N_alpha']}"
                if opt['M2N_beta'] is not None:
                    alpha_beta += f"_b{opt['M2N_beta']}"

                data_name = f"{opt['data_type']}_2d_test_{opt['mesh_type']}" \
                                   f"_{opt['fast_M2N_monitor']}_{mesh_dims[0]}_{formatted_mon_reg}reg{alpha_beta}"
            else:
                data_name = f"{opt['data_type']}_2d_test_{opt['mesh_type']}" \
                                   f"_{mesh_dims[0]}_{formatted_mon_reg}reg"

    elif opt['data_type'] == 'structured':
        num_data = opt['num_train'] if train_test == 'train' else opt['num_test']
        if data_dim == 1:
            data_name = f"{opt['data_type']}_1d_{train_test}_{num_data}_{mesh_dims[0]}_" \
                        f"{formatted_mon_reg}reg_{opt['num_gauss']}gauss_{opt['scale']}scale_{opt['mon_power']}pow"
        elif data_dim == 2:
            data_name = f"{opt['data_type']}_2d_{train_test}_{num_data}_{opt['mesh_type']}_{mesh_dims[0]}_" \
                        f"{formatted_mon_reg}reg_{opt['num_gauss']}gauss_{opt['scale']}scale_{opt['mon_power']}pow"

    elif opt['data_type'] == 'randg':
        num_data = opt['num_train'] if train_test == 'train' else opt['num_test']
        if data_dim == 1:
            if opt['num_gauss'] > 1:
                data_name = f"{opt['data_type']}_1d_{train_test}_{num_data}_{mesh_dims[0]}_{formatted_mon_reg}reg_{opt['num_gauss']}gauss"
            else:
                data_name = f"{opt['data_type']}_1d_{train_test}_{num_data}_{mesh_dims[0]}_{formatted_mon_reg}reg"

        elif data_dim == 2:
            if opt['num_gauss'] > 1:
                data_name = f"{opt['data_type']}_2d_{train_test}_{num_data}_{opt['mesh_type']}_{mesh_dims[0]}_{formatted_mon_reg}reg_{opt['num_gauss']}gauss"
            else:
                data_name = f"{opt['data_type']}_2d_{train_test}_{num_data}_{opt['mesh_type']}_{mesh_dims[0]}_{formatted_mon_reg}reg"

    elif opt['data_type'] == 'randg_mix':
        num_data = opt['num_train'] if train_test == 'train' else opt['num_test']
        if data_dim == 1:
            data_name = f"{opt['data_type']}_1d_{train_test}_{num_data}_{formatted_mon_reg}reg_{opt['num_gauss_range'][0]}_{opt['num_gauss_range'][-1]}gauss"
        elif data_dim == 2:
            data_name = f"{opt['data_type']}_2d_{train_test}_{num_data}_{formatted_mon_reg}reg_{opt['num_gauss_range'][0]}_{opt['num_gauss_range'][-1]}gauss"




    opt['data_name'] = data_name

    return opt

