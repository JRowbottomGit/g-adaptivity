import time
import pandas as pd
import numpy as np
import torch
from firedrake import VectorFunctionSpace, Function, dx, inner, assemble, sqrt, UnitIntervalMesh, UnitSquareMesh, FunctionSpace
# from matplotlib.pyplot import plot, tripcolor, triplot
from firedrake.pyplot import plot, tripcolor, triplot
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import wandb

from firedrake_difFEM.solve_poisson import poisson2d_fgauss_b0, poisson2d_fmultigauss_bcs, poisson1d_fmultigauss_bcs, plot_solutions
from firedrake_difFEM.difFEM_poisson_2d import soln
from utils_main import vizualise_grid_with_edges
from utils_data import reshape_grid_to_fd_tensor, map_firedrake_to_cannonical_ordering_2d, reshape_fd_tensor_to_grid
from firedrake_difFEM.difFEM_poisson_1d import Fixed_Mesh_1D, backFEM_1D, torch_FEM_1D, u_true_exact_1d
from firedrake_difFEM.difFEM_poisson_2d import torch_FEM_2D, u_true_exact_2d

from data_mixed_loader import Mixed_DataLoader

def evaluate_error(uu, u_true):
    ''' Calculate L1 and L2 error norms of the approximation against the exact solution '''

    L2_error = sqrt(assemble(inner(uu - u_true, uu - u_true) * dx))
    L1_error = assemble(abs(uu - u_true) * dx)

    return L1_error, L2_error

def evaluate_error_np(uu, u_true, x):
    # Calculate the lengths of each interval
    dx = np.diff(x)

    # Calculate local errors
    local_L2_errors = ((uu - u_true) ** 2)[1:] + ((uu - u_true) ** 2)[:-1]
    local_L1_errors = np.abs(uu - u_true)[1:] + np.abs(uu - u_true)[:-1]

    # Apply the trapezium rule to calculate global error
    L2_error = np.sqrt(np.sum(local_L2_errors * dx) / 2)
    L1_error = np.sum(local_L1_errors * dx) / 2

    return L1_error, L2_error

def evaluate_error_np_2d(uu, u_true, x):
    #x=np.transpose(x1)
    # Calculate the lengths of each interval in both dimensions
    dx = np.diff(x[0], axis=1)[:-1,:]
    dy = np.diff(x[1], axis=0)[:,:-1]
    uu = uu.reshape(x[0].shape)
    u_true = u_true.reshape(x[0].shape)
    error=uu-u_true

    # Calculate local errors
    local_L2_errors = (error ** 2)[:-1, 1:] + (error ** 2)[1:, :-1]
    local_L2_errors += (error ** 2)[1:, 1:] + (error ** 2)[:-1, :-1]
    local_L1_errors = np.abs(error)[:-1, 1:] + np.abs(error)[1:, :-1]
    local_L1_errors += np.abs(error)[1:, 1:] + np.abs(error)[:-1, :-1]

    # Apply the trapezium rule to calculate global error
    L2_error = np.sqrt(np.sum(local_L2_errors * dx * dy) / 4)
    L1_error = np.sum(local_L1_errors * dx * dy) / 4

    return L1_error, L2_error


def calculate_error_reduction(e_initial, e_adapted):
    ''' Calculate the percentage of error reduction '''
    if e_adapted == 0.:
        return None
    else:
        return (e_adapted - e_initial) / e_initial * 100


def update_mesh_coords(mesh, new_coords):
    V = VectorFunctionSpace(mesh, 'P', 1)
    new_coordinates = Function(V)
    dim = len(new_coords.shape)
    if dim == 1 or new_coords.shape[1] == 1:
        new_coords = new_coords.squeeze()
        new_coordinates.dat.data[:] = new_coords
    elif dim == 2:
        new_coordinates.dat.data[:] = new_coords
    mesh.coordinates.assign(new_coordinates)


def firedrake_call_fct_handler(fd_fct, point_list):
    '''to handle this unresolved firedrake issue:
    https://github.com/firedrakeproject/firedrake/issues/2359
    ie firedrake.function.PointNotInDomainError: domain <Mesh #2> does not contain point [0.6]
    nb other option is changing tolerance:
                #uu_MA_coarse.function_space().mesh().coordinates.dat.data_ro
            #{PointNotInDomainError}domain <Mesh #2> does not contain point [0.34 0.  ]
            # uu_MA_coarse.at(np.array([0.34,0.]), tolerance=0.1)
    '''
    fd_fct_eval = np.array(fd_fct.at(point_list, dont_raise=True))
    # extract indices where non
    non_idxs = np.where(fd_fct_eval == None)
    print(f"Number of bad eval mesh points: {len(non_idxs)} ie {len(non_idxs) / len(fd_fct_eval)}")
    fd_fct_eval[non_idxs] = 0.

    return fd_fct_eval, non_idxs


def evaluate_model_fine(model, dataset, opt, fine_eval=True):
    #evaluation on a finer mesh using opt['eval_quad_points']
    dim = len(dataset.opt['mesh_dims'])

    # #set eval_fct
    if dim == 1:
        eval_fct = poisson1d_fmultigauss_bcs
        if fine_eval:
            eval_mesh = UnitIntervalMesh(opt['eval_quad_points'] - 1, name="eval_mesh")
    elif dim == 2:
        eval_fct = poisson2d_fmultigauss_bcs
        if fine_eval:
            eval_mesh = UnitSquareMesh(opt['eval_quad_points'] - 1, opt['eval_quad_points'] - 1, name="eval_mesh")
            x_values = np.linspace(0, 1, opt['eval_quad_points'])
            y_values = np.linspace(0, 1, opt['eval_quad_points'])
            X, Y = np.meshgrid(x_values, y_values)
            eval_grid = [X, Y]
            eval_vec = np.reshape(np.array([X, Y]), [2, opt['eval_quad_points']**2])

    if opt['data_type'] == 'randg_m2n':# and opt['batch_size'] > 1:
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False, exclude_keys=exclude_keys, follow_batch=follow_batch)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results, times = [], []
    for i, data in enumerate(loader):
        if opt['overfit_num']:
            if i not in opt['overfit_num']:
                continue  # skip to next batch
            else:
                print(f"Overfitting on batch {i} of {opt['overfit_num']}")

        if opt['data_type'] == 'randg_m2n':
            mesh = data.mesh[0]
            deformed_mesh = data.mesh_deformed[0]
            c_list = data.batch_dict[0]['pde_params']['centers']
            s_list = data.batch_dict[0]['pde_params']['scales']
            mapping_tensor = data.mapping_tensor
        else:
            mesh = dataset.mesh
            deformed_mesh = dataset.mesh_deformed
            c_list = data.pde_params['centers'][0]
            s_list = data.pde_params['scales'][0]
            mapping_tensor = dataset.mapping_tensor

        if dim == 1:
            num_meshpoints = dataset.opt['mesh_dims'][0]
        elif dim == 2:
            num_meshpoints = int(np.sqrt(data.x_phys.shape[0]))
            if not fine_eval:
                eval_mesh = mesh
                x_values = np.linspace(0, 1, num_meshpoints)
                y_values = np.linspace(0, 1, num_meshpoints)
                X, Y = np.meshgrid(x_values, y_values)
                eval_grid = [X, Y]
                eval_vec = np.reshape(np.array([X, Y]), [2, num_meshpoints**2])

        data.idx = i

        if fine_eval:
            try: #from data_all.py and data_all_fine.py do this
                if opt['data_type'] == 'randg_m2n':
                    eval_errors = data.batch_dict[0]['eval_errors']
                    L1_grid, L2_grid = eval_errors['L1_grid'].item(), eval_errors['L2_grid'].item()
                    L1_MA, L2_MA = eval_errors['L1_MA'].item(), eval_errors['L2_MA'].item()
                else:
                    L1_grid, L2_grid = data.eval_errors['L1_grid'].item(), data.eval_errors['L2_grid'].item()
                    L1_MA, L2_MA = data.eval_errors['L1_MA'].item(), data.eval_errors['L2_MA'].item()
                MA_time = data.build_time.item()
            except:
                print("Pre process eval data doesn't exists, calculating...")
                if dim == 1:
                    fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, mesh, deformed_mesh, eval_mesh, eval_fct, dim, num_meshpoints, c_list, s_list, opt)
                elif dim == 2:
                    fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, mesh, deformed_mesh, eval_mesh, eval_fct, dim, num_meshpoints, c_list, s_list, opt, eval_vec, X, Y)
        else:
            if dim == 1:
                fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, mesh, deformed_mesh, eval_mesh, eval_fct, dim, num_meshpoints, c_list, s_list, opt)
            elif dim == 2:
                fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, mesh, deformed_mesh, eval_mesh, eval_fct, dim, num_meshpoints, c_list, s_list, opt, eval_vec, X, Y)

            L1_grid, L2_grid = eval_errors_dict['L1_grid'], eval_errors_dict['L2_grid']
            L1_MA, L2_MA = eval_errors_dict['L1_MA'], eval_errors_dict['L2_MA']

        #3) Get the model deformed mesh from trained model
        start_MLmodel = time.time()
        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'modular':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss':
            coeffs, MLmodel_coords, sol = model(data)
            MLmodel_coords = MLmodel_coords.to('cpu').detach().numpy()
        MLmodel_time = model.end_MLmodel - start_MLmodel

        MLmodel_coords = MLmodel_coords.squeeze() if dim == 1 else MLmodel_coords
        update_mesh_coords(mesh, MLmodel_coords)

        if not fine_eval:
            eval_mesh = mesh
            # eval_grid = MLmodel_coords
            #make eval grid in the style of mesh grid using MLmodel_coords using repeat
            if dim == 2:
                mesh_dims = [num_meshpoints, num_meshpoints]
                torch_MLmodel_coords = torch.tensor(MLmodel_coords)
                eval_grid0 = reshape_fd_tensor_to_grid(torch_MLmodel_coords[:, 0], mapping_tensor, mesh_dims, batch_size=1, dim=2).to('cpu').detach().numpy()
                eval_grid1 = reshape_fd_tensor_to_grid(torch_MLmodel_coords[:, 1], mapping_tensor, mesh_dims, batch_size=1, dim=2).to('cpu').detach().numpy()
                eval_grid = np.concatenate((eval_grid0, eval_grid1), axis=0)

        # solve PDE on either fine grid with firedrake or analytically
        if opt['evaler'] == 'fd_*':
            uu_MLmodel_fine, u_true_MLmodel_fine, f_MLmodel_fine = eval_fct(eval_mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
            u_MLmodel_eval_ref = uu_MLmodel_fine
        elif opt['evaler'] == 'analytical':
            if dim == 1:
                u_MLmodel_eval_ref = u_true_exact_1d(torch.from_numpy(eval_mesh.coordinates.dat.data_ro), c_list, s_list).to('cpu').detach().numpy()
            if dim == 2:
                u_MLmodel_eval_ref = u_true_exact_2d(torch.tensor(eval_grid), c_list, s_list).to('cpu').detach().numpy()

        #evaluate error
        if dim == 1:
            uu_MLmodel, u_true_MLmodel, f_MLmodel, L1_MLmodel, L2_MLmodel, L1_MLmodel_fd, L2_MLmodel_fd \
                    = solve_eval_1data(mesh, eval_mesh, u_MLmodel_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, fine_eval=fine_eval)
        elif dim == 2:
            uu_MLmodel, u_true_MLmodel, f_MLmodel, L1_MLmodel, L2_MLmodel, L1_MLmodel_fd, L2_MLmodel_fd \
                    = solve_eval_1data(mesh, eval_mesh, u_MLmodel_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, X, Y, fine_eval=fine_eval)

        # Calculate error reduction ratios
        L1_reduction_MA = calculate_error_reduction(L1_grid, L1_MA)
        L2_reduction_MA = calculate_error_reduction(L2_grid, L2_MA)
        L1_reduction_MLmodel = calculate_error_reduction(L1_grid, L1_MLmodel)
        L2_reduction_MLmodel = calculate_error_reduction(L2_grid, L2_MLmodel)

        results.append({
            'L1_grid': L1_grid,
            'L2_grid': L2_grid,
            'L1_MA': L1_MA,
            'L2_MA': L2_MA,
            'L1_MLmodel': L1_MLmodel,
            'L2_MLmodel': L2_MLmodel,
            'L1_reduction_MA': L1_reduction_MA,
            'L2_reduction_MA': L2_reduction_MA,
            'L1_reduction_MLmodel': L1_reduction_MLmodel,
            'L2_reduction_MLmodel': L2_reduction_MLmodel
        })

        times.append({
            'MA_time': MA_time,
            'MLmodel_time': MLmodel_time})


    df = pd.DataFrame(results)
    df_time = pd.DataFrame(times)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.describe())
    print(df_time.describe())

    return df, df_time


def eval_grid_MMPDE_MA(data, x_comp_mesh, x_phys_mesh, eval_mesh, eval_fct, dim, num_meshpoints, c_list, s_list, opt, eval_vec=None, X=None, Y=None, fine_eval=True):

    # 1) Get the non-deformed mesh solution
    if fine_eval:
        update_mesh_coords(x_comp_mesh, data.x_comp)
    else:
        update_mesh_coords(eval_mesh, data.x_comp)

    if opt['evaler'] == 'fd_*':
        #here u_grid_eval_ref is a firedrake function
        uu_grid_fine, u_true_grid_fine, _, _ = eval_fct(eval_mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
        u_grid_eval_ref = uu_grid_fine
    elif opt['evaler'] == 'analytical':
        #here u_grid_eval_ref is actually a canon vec of the true solution on the fine mesh
        if dim == 1:
            u_grid_eval_ref = u_true_exact_1d(torch.from_numpy(eval_mesh.coordinates.dat.data_ro), c_list, s_list).to('cpu').detach().numpy()
        elif dim == 2:
            u_grid_eval_ref = u_true_exact_2d(torch.tensor(eval_vec), c_list, s_list).to('cpu').detach().numpy()

    if dim == 1:
        uu_grid, u_true_grid, f_grid, L1_grid, L2_grid, L1_grid_fd, L2_grid_fd \
                    = solve_eval_1data(x_comp_mesh, eval_mesh, u_grid_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, fine_eval=fine_eval)
    elif dim == 2:
        uu_grid, u_true_grid, f_grid, L1_grid, L2_grid, L1_grid_fd, L2_grid_fd \
                    = solve_eval_1data(x_comp_mesh, eval_mesh, u_grid_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, X, Y, fine_eval=fine_eval)

    # 2) Get the MA-MMPDE5 deformed mesh solution
    if fine_eval:
        update_mesh_coords(x_phys_mesh, data.x_phys)
        if dim == 1:
            eval_grid = data.x_phys
        elif dim == 2:
            eval_grid = np.array([X, Y])

    else:
        update_mesh_coords(eval_mesh, data.x_phys)
        if dim == 1:
            eval_grid = data.x_phys
        elif dim == 2:
            mapping_tensor = data.mapping_tensor
            mesh_dims = [num_meshpoints, num_meshpoints]
            x_phys_coords = torch.tensor(data.x_phys)
            eval_grid0 = reshape_fd_tensor_to_grid(x_phys_coords[:, 0], mapping_tensor, mesh_dims, batch_size=1, dim=2).to('cpu').detach().numpy()
            eval_grid1 = reshape_fd_tensor_to_grid(x_phys_coords[:, 1], mapping_tensor, mesh_dims, batch_size=1, dim=2).to('cpu').detach().numpy()
            eval_grid = np.array([eval_grid0, eval_grid1])

    if opt['evaler'] == 'fd_*':
        u_MA_eval_ref = uu_grid_fine

    elif opt['evaler'] == 'analytical':
        if dim == 1:
            u_MA_eval_ref = u_true_exact_1d(torch.from_numpy(eval_mesh.coordinates.dat.data_ro), c_list, s_list).to('cpu').detach().numpy()
        if dim == 2:
            u_MA_eval_ref = u_true_exact_2d(torch.tensor(eval_grid), c_list, s_list).to('cpu').detach().numpy()

    try:
        if dim == 1:
            uu_ma, u_true_ma, f_ma, L1_ma, L2_ma, L1_ma_fd, L2_ma_fd \
                        = solve_eval_1data(x_phys_mesh, eval_mesh, u_MA_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, fine_eval=fine_eval)
        elif dim == 2:
            uu_ma, u_true_ma, f_ma, L1_ma, L2_ma, L1_ma_fd, L2_ma_fd \
                        = solve_eval_1data(x_phys_mesh, eval_mesh, u_MA_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, X, Y, fine_eval=fine_eval)
    except:
        print("Error in solving MA-MMPDE5")
        uu_ma, u_true_ma, f_ma, L1_ma, L2_ma, L1_ma_fd, L2_ma_fd = 0., 0., 0., 0., 0., 0., 0.

    fct_on_grids_dict = {
        'uu_grid': uu_grid,
        'u_true_grid': u_true_grid,
        'f_grid': f_grid,
        'uu_ma': uu_ma,
        'u_true_ma': u_true_ma,
        'f_ma': f_ma}

    eval_errors_dict = {
        'L1_grid': L1_grid,
        'L2_grid': L2_grid,
        'L1_MA': L1_ma,
        'L2_MA': L2_ma,
        'L1_grid_fd': L1_grid_fd,
        'L2_grid_fd': L2_grid_fd,
        'L1_MA_fd': L1_ma_fd,
        'L2_MA_fd': L2_ma_fd
                    }

    return fct_on_grids_dict, eval_errors_dict


def solve_eval_1data(mesh, eval_mesh, u_eval_ref, eval_fct, dim, num_meshpoints, c_list, s_list, opt, X=None, Y=None, fine_eval=True):
    # Interpolate solution on "any" coarse mesh onto the (fine) eval_mesh and evaluate against "u_eval_ref" to give L1 L2 errors

    #given a mesh get uu/u_true/f at mesh points using eval_fct
    if dim == 1:
        uu_coarse, u_true_coarse, f_coarse = eval_fct(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
    elif dim == 2:
        uu_coarse, u_true_coarse, f_coarse = eval_fct(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)

    #project onto fine eval_mesh and calculate L1 and L2 errors
    if opt['solver'] == 'firedrake':
        L1_fd, L2_fd = evaluate_error(uu_coarse, u_true_coarse)
        #evaluate the firedrake function on the firedrake fine mesh coords
        uu_coarse2fine, non_idxs = firedrake_call_fct_handler(fd_fct=uu_coarse, point_list=eval_mesh.coordinates.dat.data_ro.tolist())

        if dim == 1:
            u_eval_ref[non_idxs] = 0.
            L1, L2 = evaluate_error_np(u_eval_ref, uu_coarse2fine, eval_mesh.coordinates.dat.data_ro)
        elif dim == 2:
            if opt['evaler'] == 'fd_*':
                raise NotImplementedError
                # u_eval_ref is firedrake function on eval mesh
                # uu_coarse2fine is a np array in firedrake ordering
                # needed to reorder uu_coarse2fine from fd to grid which needs mapping_tensor
                # uu_coarse2fine_grid = reshape_fd_tensor_to_grid(uu_coarse2fine, mapping_tensor, mesh_dims, batch_size=1, dim=2).to('cpu').detach().numpy()
            elif opt['evaler'] == 'analytical': #get canon true solution
                # u_eval_ref is a conan ordered fine grid
                #but uu_coarse2fine is in firedrake ordering
                raise NotImplementedError
                # L1, L2 = evaluate_error_np_2d(u_eval_ref, uu_coarse2fine, np.array([X, Y]))
        # return uu_coarse, u_true_coarse, f_coarse, L1, L2, L1_fd, L2_fd

    elif opt['solver'] == 'torch_FEM':
        c_list = [torch.from_numpy(c_0) for c_0 in c_list]
        s_list = [torch.from_numpy(s_0) for s_0 in s_list]
        mesh_points = torch.from_numpy(mesh.coordinates.dat.data_ro).float()
        if dim == 1:
            quad_points = torch.from_numpy(eval_mesh.coordinates.dat.data_ro).float()
            _, _, uu_coarse2fine, _, _ = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list)
            L1, L2 = evaluate_error_np(uu_coarse2fine.to('cpu').detach().numpy(), u_eval_ref, eval_mesh.coordinates.dat.data_ro)
        elif dim == 2:
            if opt['evaler'] == 'fd_*':
                raise NotImplementedError
                # u_eval_ref is firedrake function on eval mesh
            eval_grid_np = np.array([X, Y])
            eval_grid = torch.tensor(eval_grid_np)
            _, _, uu_coarse2fine = torch_FEM_2D(opt, mesh, mesh_points, eval_grid, num_meshpoints, c_list, s_list)
            L1, L2 = evaluate_error_np_2d(uu_coarse2fine.to('cpu').detach().numpy(), u_eval_ref, eval_grid_np)

    #todo could include analytical solution here as well

    return uu_coarse, u_true_coarse, f_coarse, L1, L2, None, None


def linear_interpolate_FD(coarse_function, fine_mesh, dim):
    """
    Interpolate 1D data from a coarse mesh to a fine mesh.
    :param coarse_values: The values on the coarse mesh (1D numpy array).
    :param fine_mesh_size: The number of points in the fine mesh.
    :return: Interpolated values on the fine mesh (1D numpy array).
    """
    if dim == 1:
        coarse_values = coarse_function.vector().get_local()
        coarse_mesh_size = len(coarse_values)
        fine_mesh_size = fine_mesh.coordinates.dat.shape[0]
        x_coarse = np.linspace(0, 1, coarse_mesh_size)
        x_fine = np.linspace(0, 1, fine_mesh_size)
        fine_values = np.interp(x_fine, x_coarse, coarse_values)
        # fine_mesh = UnitIntervalMesh(fine_mesh_size)
        fine_function = Function(FunctionSpace(fine_mesh, 'CG', 1))
        fine_function.vector().set_local(fine_values)
    else:
        # Interpolate coarse solution onto the fine mesh
        V_fine = FunctionSpace(fine_mesh, 'CG', 1)
        fine_function = Function(V_fine)
        fine_function.interpolate(coarse_function, allow_missing_dofs=True)

    return fine_function

def linear_interpolate_np(coarse_values, fine_mesh_size):
    """
    Interpolate 1D data from a coarse mesh to a fine mesh.
    :param coarse_values: The values on the coarse mesh (1D numpy array).
    :param fine_mesh_size: The number of points in the fine mesh.
    :return: Interpolated values on the fine mesh (1D numpy array).
    """
    coarse_mesh_size = len(coarse_values)
    x_coarse = np.linspace(0, 1, coarse_mesh_size)
    x_fine = np.linspace(0, 1, fine_mesh_size)
    fine_values = np.interp(x_fine, x_coarse, coarse_values)
    return fine_values

def loop_model_output(model, mesh, dataset, opt):
    # Create a DataLoader with batch size of 1 to load one data point at a time
    loader = DataLoader(dataset, batch_size=1)

    # Create a list to store the model outputs
    model_outputs = []

    for i, data in enumerate(loader):
        data.idx = i
        c_list = data.pde_params['centers'][0]
        s_list = data.pde_params['scales'][0]
        c = c_list[0][0]
        s = s_list[0][0]

        if opt['solver'] == 'firedrake':
            uu_grid_coarse, u_true_grid_coarse, _, _ = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
        elif opt['solver'] == 'torch_FEM':
            mesh_points = torch.from_numpy(mesh.coordinates.dat.data_ro).float()
            num_meshpoints = mesh_points.shape[0]
            #todo should be sending QUAD not mesh_points here?
            uu_grid_coeffs, _, uu_grid_sol, BC1, BC2  = torch_FEM_1D(opt, mesh_points, mesh_points, num_meshpoints, c, s)
            # uu_grid_coarse = uu_grid_coarse.to('cpu').detach().numpy()
            # full_sol = torch.cat((BC1, uu_grid_coeffs, BC2), 0)
            # uu_grid_coarse = full_sol.to('cpu').detach().numpy()

        # Append the model outputs to the list
        model_outputs.append(uu_grid_sol)

    return model_outputs


def plot_trained_dataset_1d(dataset, model, opt, model_out=None, show_mesh_evol_plots=False):
    mesh = dataset.mesh
    dim = len(dataset.opt['mesh_dims'])
    num_meshpoints = dataset.opt['mesh_dims'][0] if dim == 1 else dataset.opt['mesh_dims'][0] * dataset.opt['mesh_dims'][1]
    fine_mesh = UnitIntervalMesh(opt['eval_quad_points'] - 1, name="fine_mesh")

    # Create a DataLoader with batch size of 1 to load one data point at a time
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

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

    # figure for FEM on MLModel mesh
    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))  # adjust as necessary
    axs2 = axs2.ravel()
    fig2.suptitle('FEM on MLmodel mesh', fontsize=20)
    fig2.tight_layout()

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
        # todo check this annoying property of PyG I believe, making indexing necessary
        #this happens for numpy arrays in PyG datasets
        c_list = data.pde_params['centers'][0]
        s_list = data.pde_params['scales'][0]

        #gen fine baseline true solution
        uu_fine, u_true_fine, _ = poisson1d_fmultigauss_bcs(fine_mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)

        # 1) plot the FEM on regular mesh
        plot(data.uu[0], axes=axs0[i], label='uu_fem_xcomp', color='orange')
        plot(data.u_true[0], axes=axs0[i], label='u_true_xcomp', color='green')
        plot(uu_fine, axes=axs0[i], label='uu_fem_fine', color='lightblue')
        plot(u_true_fine, axes=axs0[i], label='u_true_fine', color='grey')

        if opt['solver'] == 'firedrake':
            uuFD_coarse, u_true_FD_coarse, _ = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
            plot(uuFD_coarse, axes=axs0[i], label='FD_out', color='purple')

        elif opt['solver'] == 'torch_FEM':
            c_list_torch = [torch.from_numpy(c_0) for c_0 in c_list]
            s_list_torch = [torch.from_numpy(s_0) for s_0 in s_list]
            mesh_points = torch.from_numpy(mesh.coordinates.dat.data_ro).float()
            quad_points = torch.from_numpy(fine_mesh.coordinates.dat.data_ro).float()
            UUtorchSolcoeffs, mesh_points, uutorch_coarse2fine, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list_torch, s_list_torch)
            full_UUtorchsol = torch.cat((BC1, UUtorchSolcoeffs.squeeze(), BC2), 0)
            axs0[i].plot(mesh_points, full_UUtorchsol, label='torchFEM_out', color='purple')


        #scatter plot of data.u_true on the regular mesh
        # axs0[i].scatter(data.x_comp, data.u_true[0].dat.data_ro, color='red', marker='x', label='u_true_x_comp')
        #nb u_true from poisson1d_fmultigauss_bcs is the projection of the true solution onto FEM basis
        #this solves the variational problem for the true solution and can be inaccurate at low resolution
        #there call
        u_true_exact_1d_vals = u_true_exact_1d(data.x_comp, c_list, s_list).to('cpu').detach().numpy()
        axs0[i].scatter(data.x_comp, u_true_exact_1d_vals, color='red', marker='x', label='u_true_x_comp')

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
            uu_MA_coarse, u_true_MA_coarse, _ = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
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

        #scatter plot of data.u_true on the regular mesh
        # axs1[i].scatter(data.x_comp, data.u_true[0].dat.data_ro, color='red', marker='x', label='u_true_x_comp')
        # u_true_exact_1d = u_true_exact_1d(data.x_comp, c, s).to('cpu').detach().numpy()
        axs1[i].scatter(data.x_comp, u_true_exact_1d_vals, color='red', marker='x', label='u_true_x_comp')

        #scatter plot of data.u_true on the MA mesh
        # axs1[i].scatter(data.x_phys, u_true_ma.dat.data_ro, color='blue', marker='x', label='u_true_MA')
        u_true_exact_1d_xphys = u_true_exact_1d(data.x_phys, c_list, s_list).to('cpu').detach().numpy()
        axs1[i].scatter(data.x_phys, u_true_exact_1d_xphys, color='blue', marker='x', label='u_true_MA')

        #x-axis dashed for the x_phys
        extraticks = data.x_phys.tolist()
        # axs1[i].set_xticks(list(axs1[i].get_xticks()) + extraticks)
        for tick in extraticks:
            axs1[i].plot([tick, tick], [ymin, ymin + dash_length], color=dashcol, linestyle='-', linewidth=dashwid)
        axs1[i].legend()

        # 3) plot the MLmodel mesh
        #do a forward pass of the model to get deformed mesh and collect intermediate states
        if opt['model'] in ['backFEM_1D', 'learn_Mon_1D', 'GNN'] and show_mesh_evol_plots:
            model.plot_evol_flag = True

        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss':
            coeffs, MLmodel_coords, sol = model(data)
            MLmodel_coords = MLmodel_coords.to('cpu').detach().numpy()

        if opt['model'] in ['backFEM_1D', 'learn_Mon_1D', 'GNN'] and show_mesh_evol_plots:
            model.plot_evol_flag = False

        # MLmodel_coords = model(data).to('cpu').detach().numpy()
        mesh = dataset.mesh
        update_mesh_coords(mesh, MLmodel_coords)
        # solve poisson
        uu_MLmodel, u_true_MLmodel, f = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
        # colors = tripcolor(uu_gnn, axes=axs4[i])#, shading='gouraud', cmap='viridis')
        plot(uu_MLmodel, axes=axs2[i], label='uu_fem_MLmodel', color='orange')
        plot(u_true_MLmodel, axes=axs2[i], label='u_true_MLmodel', color='green')
        plot(uu_fine, axes=axs2[i], label='uu_fem_fine', color='lightblue')
        plot(u_true_fine, axes=axs2[i], label='u_true_fine', color='grey')

        #scatter plot of data.u_true on the regular mesh
        # axs2[i].scatter(data.x_comp, data.u_true[0].dat.data_ro, color='red', marker='x', label='u_true_x_comp')
        axs2[i].scatter(data.x_comp, u_true_exact_1d_vals, color='red', marker='x', label='u_true_x_comp')

        #scatter plot of data.u_true on the updated mesh
        # axs2[i].scatter(MLmodel_coords, u_true_MLmodel.dat.data_ro, color='blue', marker='x', label='u_true_MLmodel')
        u_true_exact_1d_MLmodel_vals = u_true_exact_1d(MLmodel_coords.squeeze(), c_list, s_list)
        axs2[i].scatter(MLmodel_coords, u_true_exact_1d_MLmodel_vals, color='blue', marker='x', label='u_true_MLmodel')

        #x-axis dashed for the MLmodel_coords
        extraticks = MLmodel_coords.tolist()
        for tick in extraticks:
            axs2[i].plot([tick, tick], [ymin, ymin + dash_length], color=dashcol, linestyle='-', linewidth=dashwid)
        axs2[i].legend()

    if opt['wandb_log_plots'] and show_mesh_evol_plots:
        wandb.log({'fem_on_x_comp': wandb.Image(fig0)})
        wandb.log({'fem_on_mmpde5': wandb.Image(fig1)})
        wandb.log({'fem_on_MLmodel': wandb.Image(fig2)})
        if opt['model'] in ['backFEM_1D', 'learn_Mon_1D', 'GNN']:
            wandb.log({'mesh_evol': wandb.Image(model.mesh_fig)})
        if opt['model'] in ['backFEM_1D']:
            wandb.log({'loss': wandb.Image(model.loss_fig)})

    if opt['show_plots']:
        plt.show()


def plot_trained_dataset_2d(dataset, model, opt, show_mesh_evol_plots=False):
    # Create a DataLoader with batch size of 1 to load one data point at a time
    # loader = DataLoader(dataset, batch_size=1)
    if opt['data_type'] == 'randg_m2n':# and opt['batch_size'] > 1:
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False, exclude_keys=exclude_keys, follow_batch=follow_batch)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    #figure for FEM on regular mesh
    fig0, axs0 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs0 = axs0.ravel()
    fig0.suptitle('FEM on regular mesh', fontsize=20)
    fig0.tight_layout()

    # figure for MA mesh
    fig1, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs1 = axs.ravel()
    fig1.suptitle('MA mesh', fontsize=20)
    fig1.tight_layout()

    # #figure for FEM on MA mesh
    fig2, axs2 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs2 = axs2.ravel()
    fig2.suptitle('FEM on MA mesh', fontsize=20)
    fig2.tight_layout()

    # figure for MLmodel mesh
    fig3, axs3 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs3 = axs3.ravel()
    fig3.suptitle('MLmodel mesh', fontsize=20)
    fig3.tight_layout()

    # figure for FEM on MLmodel mesh
    fig4, axs4 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs4 = axs4.ravel()
    fig4.suptitle('FEM on MLmodel mesh', fontsize=20)
    fig4.tight_layout()

    # # figure for error of MA mesh versus regular mesh
    # fig5, axs5 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    # axs5 = axs5.ravel()
    # fig5.suptitle('Error of MA mesh versus regular mesh - CURRENTLY NOT WELL DEFINED', fontsize=20)
    # fig5.tight_layout()
    #
    # # figure for error of MLmodel mesh versus regular mesh
    # fig6, axs6 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    # axs6 = axs6.ravel()
    # fig6.suptitle('Error of MLmodel mesh versus regular mesh - CURRENTLY NOT WELL DEFINED', fontsize=20)
    # fig6.tight_layout()

    # Loop over the dataset
    for i, data in enumerate(loader):
        data.idx = i
        if i == 25:
            break

        if opt['overfit_num']:
            if i not in opt['overfit_num']:
                continue  # skip to next batch
            else:
                print(f"Overfitting on batch {i} of {opt['overfit_num']}")

        # todo check this annoying property of PyG I believe, making indexing necessary
        if opt['data_type'] == 'randg_m2n':
            mesh = data.mesh[0]
            deformed_mesh = data.mesh_deformed[0]
            c_list = data.batch_dict[0]['pde_params']['centers']
            s_list = data.batch_dict[0]['pde_params']['scales']
        else:
            mesh = dataset.mesh
            deformed_mesh = dataset.mesh_deformed
            c_list = data.pde_params['centers'][0]
            s_list = data.pde_params['scales'][0]

        # 0) plot the FEM on regular mesh
        colors = tripcolor(data.uu[0], axes=axs0[i])  # , shading='gouraud', cmap='viridis')

        # Convert PyG graph to NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # 1) plot the MA mesh
        # Get node positions from the coordinates attribute in the PyG graph
        x = data.x_phys
        positions = {i: x[i].tolist() for i in range(x.shape[0])}
        nx.draw(G, pos=positions, ax=axs1[i], node_size=1, width=0.5, with_labels=False)

        # #2) plot the FEM on MA (target phys) mesh
        if opt['data_type'] == 'randg_m2n':
            mesh = data.mesh[0]
        else:
            mesh = dataset.mesh

        update_mesh_coords(mesh, x)
        uu_ma, u_true_ma, _ = poisson2d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
        colors = tripcolor(uu_ma, axes=axs2[i])#, shading='gouraud', cmap='viridis')

        #3) Get the model deformed mesh solution
        if opt['model'] in ['backFEM_2D', 'GNN'] and show_mesh_evol_plots:
            model.plot_evol_flag = True

        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'modular':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss':
            coeffs, MLmodel_coords, sol = model(data)
            MLmodel_coords = MLmodel_coords.to('cpu').detach().numpy()

        if opt['model'] in ['backFEM_2D', 'GNN'] and show_mesh_evol_plots:
            model.plot_evol_flag = False

        # 3) plot the MLmodel mesh and evol if applicable
        positions = {i: MLmodel_coords[i].tolist() for i in range(MLmodel_coords.shape[0])}
        nx.draw(G, pos=positions, ax=axs3[i], node_size=1, width=0.5, with_labels=False)

        # 4) plot the FEM on MLmodel mesh
        update_mesh_coords(mesh, MLmodel_coords)
        # solve poisson
        uu_gnn, u_true_gnn, f = poisson2d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list,
                                                                      rand_gaussians=False)
        colors = tripcolor(uu_gnn, axes=axs4[i])#, shading='gouraud', cmap='viridis')

        # 5) plot the error of MA mesh versus regular mesh
        # error = uu_ma - data.uu[0]
        # colors = tripcolor(error, axes=axs5[i])#, shading='gouraud', cmap='viridis')
        # error = Function(uu_ma.function_space())
        # error.assign(uu_ma - data.uu[0])
        # tripcolor(error, axes=axs5[i])

        # 6) plot the error of MLmodel mesh versus regular mesh
        # error = uu_gnn - data.uu[0]
        # colors = tripcolor(error, axes=axs6[i])#, shading='gouraud', cmap='viridis')
        # error = Function(uu_gnn.function_space())
        # error.assign(uu_gnn - data.uu[0])
        # tripcolor(error, axes=axs6[i])

    if opt['show_plots']:
        plt.show()




def plot_individual_meshes(dataset, model, opt):
    dim = len(opt['mesh_dims'])

    # update mesh coordinates
    # visualise first N meshes and results
    N = 1
    loader = DataLoader(dataset, batch_size=1)
    for i, data in enumerate(loader):
        print(f"visualising mesh {i}")
        if opt['show_plots']:
            vizualise_grid_with_edges(data.x_phys, data.edge_index, opt, boundary_nodes=data.boundary_nodes)
            vizualise_grid_with_edges(data.x_comp, data.edge_index, opt, boundary_nodes=data.boundary_nodes)
            learned_mesh = model(data)
            if opt['fix_boundary']:
                mask = ~data.to_boundary_edge_mask * ~data.to_corner_nodes_mask * ~data.diff_boundary_edges_mask
                edge_index = data.edge_index[:, mask]
                # need to add self loops for the corner nodes or they go to zero
                corner_nodes = torch.cat([torch.from_numpy(arr) for arr in data.corner_nodes]).repeat(2, 1)
                edge_index = torch.cat([edge_index, corner_nodes], dim=1)

            _ = vizualise_grid_with_edges(learned_mesh, edge_index, opt,
                                                     boundary_nodes=data.boundary_nodes, node_labels=False,
                                                     node_boundary_map=data.node_boundary_map, corner_nodes=data.corner_nodes, edge_weights=model.gnn_deformer.convs[-1].alphas)

        # update firedrake computational mesh with deformed coordinates
        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss':
            coeffs, MLmodel_coords, sol = model(data)
            MLmodel_coords = MLmodel_coords.to('cpu').detach().numpy()

        mesh = dataset.mesh
        update_mesh_coords(mesh, MLmodel_coords)
        # solve poisson
        c_list = data.pde_params['centers'][0] #todo check this annoying property of PyG I believe, making indexing necessary
        s_list = data.pde_params['scales'][0]

        if dim == 1:
            uu, u_true, f = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list,
                                                                  rand_gaussians=False)

        elif dim == 2:
            uu, u_true, f = poisson2d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list,
                                                                  rand_gaussians=False)
            plot_solutions(uu, u_true)

        if i == N-1:
            break
