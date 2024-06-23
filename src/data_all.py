import sys
sys.path.append('../')
import os
import numpy as np
import torch
import torch_geometric as pyg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from firedrake import UnitIntervalMesh, UnitSquareMesh, FunctionSpace, DirichletBC, Mesh, BoxMesh, triplot, tripcolor, CheckpointFile, plot #File

from utils_eval import eval_grid_MMPDE_MA
from data import firedrake_mesh_to_PyG, deform_mesh_mmpde1d, deform_mesh_mmpde2d, deform_mesh_ma2d, plot_initial_dataset_2d, plot_initial_dataset_1d
from utils_data import make_data_name, to_float32, map_firedrake_to_cannonical_ordering_2d, map_firedrake_to_cannonical_ordering_1d
from params_poisson import get_params, run_params
from firedrake_difFEM.solve_poisson import poisson1d_fmultigauss_bcs, poisson2d_fmultigauss_bcs
from firedrake_difFEM.difFEM_poisson_1d import Fixed_Mesh_1D
from firedrake_difFEM.difFEM_poisson_2d import Fixed_Mesh_2D


class AllMeshInMemoryDataset(pyg.data.InMemoryDataset):
    def __init__(self, root, mesh_dims, opt, transform=None, pre_transform=None):
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
        print(f"root {root}")
        super(AllMeshInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        custom_attributes_path = os.path.join(self.root, "processed", "custom_attributes.pt")
        print(f"custom_attributes_path {custom_attributes_path}")
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
            self.mesh = UnitIntervalMesh(self.n - 1, name="ref_mesh")
            self.mesh_deformed = UnitIntervalMesh(self.n - 1, name="deformed_mesh")
            self.fine_mesh = UnitIntervalMesh(self.opt['eval_quad_points'] - 1, name="fine_mesh")

        elif self.dim == 2:
            self.mesh = UnitSquareMesh(self.n - 1, self.m - 1, name="ref_mesh")
            self.mesh_deformed = UnitSquareMesh(self.n - 1, self.m - 1, name="deformed_mesh")
            self.fine_mesh = UnitSquareMesh(self.opt['eval_quad_points'] - 1, self.opt['eval_quad_points'] - 1, name="fine_mesh")

        with CheckpointFile(os.path.join(self.root, "processed", f"mesh.h5"), 'w') as mesh_file:
            mesh_file.save_mesh(self.mesh)
        with CheckpointFile(os.path.join(self.root, "processed", f"deformed_mesh.h5"), 'w') as mesh_file:
            mesh_file.save_mesh(self.mesh_deformed)

        self.x_comp_shared = torch.tensor(self.mesh.coordinates.dat.data_ro)
        self.x_fine_shared = torch.tensor(self.fine_mesh.coordinates.dat.data_ro)

        if self.dim == 1:
            mapping_dict, mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(self.x_comp_shared, self.n)
            mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, X_fd_vec_fine = map_firedrake_to_cannonical_ordering_1d(self.x_fine_shared, self.opt['eval_quad_points'])
        elif self.dim == 2:
            mapping_dict, mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(self.x_comp_shared, self.n, self.m)
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
            'orig_opt': self.opt,
        }
        torch.save(custom_attributes, os.path.join(self.root, "processed", "custom_attributes.pt"))

        idx = 0
        data_list = []
        num_data_dict = {1: 9, 2: 25}
        if opt['mesh_type'] == 'M2N':
            power_list = [0.2]
        else:
            # power_list = [0.01, 0.02, 0.04, 0.08, 0.16, 0.2, 0.25, 0.32, 0.5]
            power_list = [0.01, 0.02, 0.04, 0.08, 0.16, 0.2, 0.25, 0.32, 0.5]

        for s in [0.1, 0.2, 0.3, 0.4, 0.5]: #squared: [0.01, 0.04, 0.09, 0.16, 0.25]
            for mon_power in power_list:

                for i in range(num_data_dict[self.dim]):
                    successful_eval = True
                    data = firedrake_mesh_to_PyG(self.mesh)

                    c_list = []
                    s_list = []
                    if self.dim == 1:
                        x_coord = (i + 1) / (num_data_dict[self.dim] + 1)
                        c1 = np.array([x_coord]).astype(np.float32)
                        s1 = np.array([s]).astype(np.float32)
                        c_list.append(c1)
                        s_list.append(s1)
                    elif self.dim == 2:  # 25 interrior points in 0.1-0.9 grid and iterate over them plus a fixed central Gaussian
                        x_coord1 = i % 5 * 0.2 + 0.1
                        y_coord1 = i // 5 * 0.2 + 0.1
                        c1 = np.array([x_coord1, y_coord1]).astype('f')
                        s1 = np.array([s, s]).astype('f')
                        c2 = np.array([0.5, 0.5]).astype('f')
                        s2 = np.array([s, s]).astype('f')
                        c_list.append(c1)
                        s_list.append(s1)
                        c_list.append(c2)
                        s_list.append(s2)


                    pde_params = {'centers': c_list, 'scales': s_list}
                    pde_params['scale_list'] = s_list
                    if self.dim == 1:
                        pde_params['scale_value'] = s_list[0].item()  # just for naming
                    elif self.dim == 2:
                        pde_params['scale_value'] = s_list[0][0].item()  # just for naming
                    pde_params['mon_power'] = mon_power
                    pde_params['mesh_type'] = self.opt['mesh_type']
                    pde_params['mon_reg'] = self.opt['mon_reg']
                    pde_params['eval_quad_points'] = self.opt['eval_quad_points']
                    pde_params['fast_M2N_monitor'] = self.opt['fast_M2N_monitor']
                    if self.opt['M2N_alpha'] is not None:
                        pde_params['M2N_alpha'] = self.opt['M2N_alpha']
                    if self.opt['M2N_beta'] is not None:
                        pde_params['M2N_beta'] = self.opt['M2N_beta']
                    data.pde_params = pde_params #instance specific pde parameters

                    # deform mesh using MMPDE/MA
                    if self.dim == 1:
                        data.x_phys, data.ma_its, data.build_time = deform_mesh_mmpde1d(self.x_comp_shared, self.n, pde_params)
                    elif self.dim == 2:
                        if self.opt['dataset'] in ['fd_ma_2d', 'fd_M2N_2d']:
                            x_phys, data.ma_its, data.build_time = deform_mesh_ma2d(self.x_comp_shared, self.n, self.m, pde_params)
                            data.x_phys = torch.from_numpy(x_phys)
                        elif self.opt['dataset'] == 'fd_mmpde_2d':
                            data.x_phys, data.ma_its, data.build_time = deform_mesh_mmpde2d(self.x_comp_shared, self.n, self.m,pde_params)

                    num_meshpoints = opt['mesh_dims'][0] if self.dim == 1 else opt['mesh_dims'][0]

                    if self.dim == 1:
                        eval_fct = poisson1d_fmultigauss_bcs
                        # send in the deformed mesh so not to alter the original mesh
                        fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, self.mesh, self.mesh_deformed,
                                                                             self.fine_mesh, eval_fct, self.dim,
                                                                             num_meshpoints, c_list, s_list, opt, X=None, Y=None)
                    elif self.dim == 2:
                        eval_fct = poisson2d_fmultigauss_bcs
                        x_values = np.linspace(0, 1, opt['eval_quad_points'])
                        y_values = np.linspace(0, 1, opt['eval_quad_points'])
                        X, Y = np.meshgrid(x_values, y_values)
                        eval_vec = np.reshape(np.array([X, Y]), [2, opt['eval_quad_points'] ** 2])

                        # send in the deformed mesh so not to alter the original mesh
                        fcts_on_grids_dict, eval_errors_dict = eval_grid_MMPDE_MA(data, self.mesh, self.mesh_deformed,
                                                                             self.fine_mesh, eval_fct, self.dim,
                                                                             num_meshpoints, c_list, s_list, opt,
                                                                             eval_vec, X, Y)
                        if fcts_on_grids_dict['uu_ma'] == 0.:
                            successful_eval = False
                            print("Error in eval_grid_MMPDE_MA, saving None's")

                    # fine eval (saving tensors for fast pde loss)
                    if self.dim == 1:
                        uu_fine, u_true_fine, f_fine = poisson1d_fmultigauss_bcs(self.fine_mesh, c_list, s_list)
                    elif self.dim == 2:
                        uu_fine, u_true_fine, f_fine = poisson2d_fmultigauss_bcs(self.fine_mesh, c_list, s_list, rand_gaussians=False)

                    data.eval_errors = eval_errors_dict
                    data.successful_eval = successful_eval

                    #save the firedrake functions to file
                    filename_suffix = f"dim_{self.dim}_scale_{round(data.pde_params['scale_value'], 2)}" \
                                      f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{idx}_pde_data.h5"
                    pde_data_file = os.path.join(self.root, "processed", filename_suffix)

                    with CheckpointFile(pde_data_file, 'w') as pde_file:
                        pde_file.save_mesh(self.mesh)
                        pde_file.save_mesh(self.mesh_deformed)
                        if data.successful_eval:
                            pde_file.save_function(fcts_on_grids_dict['uu_grid'], name="uu")
                            pde_file.save_function(fcts_on_grids_dict['u_true_grid'], name="u_true")
                            pde_file.save_function(fcts_on_grids_dict['f_grid'], name="f")
                            pde_file.save_function(fcts_on_grids_dict['uu_ma'], name="uu_ma")
                            pde_file.save_function(fcts_on_grids_dict['u_true_ma'], name="u_true_ma")
                            pde_file.save_function(fcts_on_grids_dict['f_ma'], name="f_ma")

                    # also saving the torch tensors
                    # note we delay mapping firedrake functions to cannonical ordering to maintain consistency with the mesh and x_comp/phys
                    if data.successful_eval:
                        data.uu_tensor = torch.from_numpy(fcts_on_grids_dict['uu_grid'].dat.data)
                        data.u_true_tensor = torch.from_numpy(fcts_on_grids_dict['u_true_grid'].dat.data)
                        data.f_tensor = torch.from_numpy(fcts_on_grids_dict['f_grid'].dat.data)
                        data.uu_MA_tensor = torch.from_numpy(fcts_on_grids_dict['uu_ma'].dat.data)  # [self.mapping_tensor])
                        data.u_true_MA_tensor = torch.from_numpy(fcts_on_grids_dict['u_true_ma'].dat.data)  # [self.mapping_tensor])
                        data.f_MA_tensor = torch.from_numpy(fcts_on_grids_dict['f_ma'].dat.data)  # [self.mapping_tensor])
                        data.uu_fine_tensor = torch.from_numpy(uu_fine.dat.data)
                        data.u_true_fine_tensor = torch.from_numpy(u_true_fine.dat.data)
                        data.f_fine_tensor = torch.from_numpy(f_fine.dat.data)
                    else:
                        data.uu_tensor = torch.tensor([0.])
                        data.u_true_tensor = torch.tensor([0.])
                        data.f_tensor = torch.tensor([0.])
                        data.uu_MA_tensor = torch.tensor([0.])
                        data.u_true_MA_tensor = torch.tensor([0.])
                        data.f_MA_tensor = torch.tensor([0.])
                        data.uu_fine_tensor = torch.tensor([0.])
                        data.u_true_fine_tensor = torch.tensor([0.])
                        data.f_fine_tensor = torch.tensor([0.])

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

        # pde_data_file = os.path.join(self.root, "processed", f"{idx}_pde_data.h5")
        # Retrieve scale and mon_power from data object
        scale = round(data.pde_params['scale_value'].item(), 2)
        mon_power = round(data.pde_params['mon_power'].item(), 2)

        # Construct the filename using the scale, mon_power, and idx
        if 'mon_reg' in data.pde_params:
            mon_reg = round(data.pde_params['mon_reg'].item(), 2)
            pde_data_file = os.path.join(self.root, "processed", f"dim_{self.dim}_scale_{scale}_" \
                                                                 f"mon_{mon_power}_reg_{mon_reg}_{idx}_pde_data.h5")
        else:
            pde_data_file = os.path.join(self.root, "processed", f"dim_{self.dim}_scale_{scale}_mon_{mon_power}_{idx}_pde_data.h5")

        with CheckpointFile(pde_data_file, 'r') as pde_file:
            self.mesh = pde_file.load_mesh("ref_mesh")
            self.mesh_deformed = pde_file.load_mesh("deformed_mesh")
            if data.successful_eval:
                uu = pde_file.load_function(self.mesh, "uu")
                u_true = pde_file.load_function(self.mesh, "u_true")
                f = pde_file.load_function(self.mesh, "f")
                uu_ma = pde_file.load_function(self.mesh_deformed, "uu_ma")
                u_true_ma = pde_file.load_function(self.mesh_deformed, "u_true_ma")
                f_ma = pde_file.load_function(self.mesh_deformed, "f_ma")
            else:
                uu, u_true, f, uu_ma, u_true_ma, f_ma = 0., 0., 0., 0., 0., 0.

        if data.successful_eval:
            data.uu = uu
            data.u_true = u_true
            data.f = f
            data.uu_ma = uu_ma
            data.u_true_ma = u_true_ma
            data.f_ma = f_ma

        return data


def compare_firedrake_functions(func1, func2):
    attrs_to_compare = [
        'ufl_shape', 'ufl_element', 'ufl_domain', 'ufl_free_indices',
        'ufl_index_dimensions', 'dat', 'cell_set', 'function_space',
        'topological', 'subfunctions', 'comm'
    ]

    differences = {}
    for attr in attrs_to_compare:
        attr1 = getattr(func1, attr, None)
        attr2 = getattr(func2, attr, None)

        if attr1 != attr2:
            differences[attr] = (attr1, attr2)

    return differences


def dataset_summary(dataset):
    # For item-wise summary
    data_records = []

    for data in dataset:
        #loop though the dataset and extract the relevant data, detaching and numpying torch tensors as needed
        try:
            scale_value = data.pde_params['scale_value'].detach().item()
        except:
            scale_value = data.pde_params['scale_value']
        try:
            mon_power = data.pde_params['mon_power'].detach().item()
        except:
            mon_power = data.pde_params['mon_power']
        try:
            L1_grid = data.eval_errors['L1_grid'].detach().item()
        except:
            L1_grid = data.eval_errors['L1_grid']
        try:
            L2_grid = data.eval_errors['L2_grid'].detach().item()
        except:
            L2_grid = data.eval_errors['L2_grid']
        try:
            L1_MA = data.eval_errors['L1_MA'].detach().item()
        except:
            L1_MA = data.eval_errors['L1_MA']
        try:
            L2_MA = data.eval_errors['L2_MA'].detach().item()
        except:
            L2_MA = data.eval_errors['L2_MA']
        try:
            ma_its = data.ma_its.detach().item()
        except:
            try:
                ma_its = data.ma_its
            except:
                ma_its = None
        record = {
            'scale_value': scale_value, 'mon_power': mon_power,
            'L1_grid': L1_grid, 'L2_grid': L2_grid, 'L1_MA': L1_MA, 'L2_MA': L2_MA,
            'ma_its': ma_its}

        data_records.append(record)

    itemwise_df = pd.DataFrame(data_records)

    # For aggregated summary
    aggregate_functions = {
        'L1_grid': 'mean',
        'L2_grid': 'mean',
        'L1_MA': 'mean',
        'L2_MA': 'mean',
        'ma_its': 'mean'
    }
    #group by scale and mon_power and aggregate the other columns
    aggregated_df = itemwise_df.groupby(['scale_value', 'mon_power']).agg(aggregate_functions).reset_index()

    return itemwise_df, aggregated_df


# also plot data.ma_its for each scale/mon_power
def plot_scale_v_mon_its(aggregated_df, opt):
    fig, ax = plt.subplots()
    # for each scale
    for scale in aggregated_df['scale_value'].unique():
        # get the subset of the data
        subset = aggregated_df[aggregated_df['scale_value'] == scale]
        # plot the mon_power vs norm_MA
        ax.plot(subset['mon_power'], subset['ma_its'], label=scale)
    ax.legend()#"scale_value")
    #add axis and legend titles
    ax.set_xlabel('mon_power')
    ax.set_ylabel('ma_its')
    #y axis log base 10 scale
    ax.set_yscale('log')
    ax.set_title(f"ma_its vs mon_power for each scale \n {opt['data_name']}")
    plt.show()


# plot some statistics for each scale/mon_power what is the mean (over 25 centres) error change. expect 5 lines of 5 points.
def plot_scale_v_mon(aggregated_df, opt, norm="L1", dim=1, ax0=None):
    if ax0 is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = ax0
    # for each scale
    for scale in aggregated_df['scale_value'].unique():
        # get the subset of the data
        subset = aggregated_df[aggregated_df['scale_value'] == scale]
        # plot the mon_power vs norm_MA
        ax.plot(subset['mon_power'], subset[f"{norm}_MA"], label=round(scale, 2))

    ax.set_xlabel('monitor power')
    ax.set_ylabel(f"{norm} error")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=5))
    ax.set_yscale('log')

    if ax0 is None:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(bottom=0.18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    meth = "mmpde5" if dim == 1 else "MA"
    ax.set_title(f"{norm} error for {meth} monitor power vs Gauss scale - {dim}D")
    # ax.set_title(f"{norm}_MA vs mon_power for each scale \n {opt['data_name']}")
    if ax0 is None:
        plt.savefig(f"../data/{opt['data_name']}_scale_v_mon_{norm}_{meth}.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
    return ax

def plot_datasets_for_scale_and_power(dataset, scale, mon_power):
    #create mask for the dataset
    mask = (dataset.data.pde_params['scale_value'] == scale) & (dataset.data.pde_params['mon_power'] == mon_power)
    #get the subset of the dataset
    subset = dataset[mask]
    plot_initial_dataset_2d(subset)


if __name__ == '__main__':
    opt = get_params()
    if not opt['wandb_sweep']:
        opt = run_params(opt)
    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed

    data_dim = 2
    if True:
        if data_dim == 1:  # #1D
            opt['data_type'] = "all"  # "#"all" #"randg"# "all" #'randg', 'randg_m2n'
            opt['mesh_type'] = "mmpde"  #
            opt['dataset'] = f"fd_{opt['mesh_type']}_1d"
            opt['mesh_dims'] = [15]  # 10] #6]#11]21]51]101]#in terms of total number of nodes
            opt['mon_reg'] = 0.1  # 1. #0.10#0.01

        elif data_dim == 2:  # 2D
            opt['data_type'] = "all"  # "randg_m2n"#"asym_sweep2d"#"randg" #"randg"# "all"#"randg"# "all" #'randg', 'randg_m2n'
            opt['mesh_type'] = "ma"  # ma or mmpde or M2N
            if opt['mesh_type'] == "M2N":
                opt['fast_M2N_monitor'] = "slow"  # "slow" #"superslow" "fast"
                opt['M2N_alpha'] = 1.0
                opt['M2N_beta'] = 1.0
            opt[
                'dataset'] = f"fd_{opt['mesh_type']}_2d"  # 'fd_ma_grid_2d' #'fd_ma_L'#'fd_noisey_grid' #fd_ma_grid#'noisey_grid'
            opt['mesh_dims'] = [7, 7]#[11, 11]  # [15, 15] #[11, 11]
            opt['mon_reg'] = 0.01  # .1 #0.1 #0.01
        opt = make_data_name(opt)
        dataset = AllMeshInMemoryDataset(f"../data/{opt['data_name']}", opt['mesh_dims'], opt)

    make_plots = False
    if make_plots:
        fig, axes = plt.subplots(3, 2, figsize=(12, 16))
        axes = axes.flatten()
        data_dims = [1, 2] #1 #2
        opt['data_type'] = "all"
        for i, data_dim in enumerate(data_dims):
                regs = [1.0, 0.1, 0.01]
                for j, mon_reg in enumerate(regs):
                    if data_dim == 1:  # #1D
                        opt['mesh_type'] = "mmpde"
                        opt['mesh_dims'] = [11]
                    elif data_dim == 2:  # 2D
                        opt['data_type'] = "all"
                        opt['mesh_type'] = "ma"
                        opt['mesh_dims'] = [11, 11]

                    print(f"i {i} data_dim {data_dim} j {j} mon_reg {mon_reg}")
                    opt['mon_reg'] = mon_reg
                    opt = make_data_name(opt, train_test="test")
                    dataset = AllMeshInMemoryDataset(f"../data/{opt['data_name']}", opt['mesh_dims'], opt)
                    itemwise, aggregated = dataset_summary(dataset)
                    pd.set_option('display.max_columns', 10000)
                    pd.set_option('display.max_rows', 10000)
                    pd.set_option('display.width', 10000)
                    # print(itemwise)
                    print(aggregated)

                    # plot_scale_v_mon(aggregated, opt, norm="L1")
                    plot_scale_v_mon(aggregated, opt, norm="L2", dim=data_dim, ax0=axes[i * 3 + j])
                    # plot_scale_v_mon_its(aggregated, opt)

                #plot the intermediate plot whilst still building the main  plot
                # plt.show()

        # Create a single legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Scale", loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5)

        # Adjust layout
        plt.tight_layout(rect=[0, 0., 1, 0.95])
        plt.subplots_adjust(bottom=0.1)

        # Save the plot as a PDF
        plt.savefig(f"../data/{opt['data_name']}_scale_v_mon_combined.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.show()


    #2D
    #i) MMPDE or MA
    mask = (dataset.data.pde_params['scale_value'] == 0.1) & (dataset.data.pde_params['mon_power'] == 0.2)#2)
    dataset = dataset[mask]

    if dataset.dim == 1:
        opt['model'] = 'fixed_mesh_1D'
        model = Fixed_Mesh_1D(opt)
        plot_initial_dataset_1d(dataset, opt)
    elif dataset.dim == 2:
        opt['model'] = 'fixed_mesh_2D'
        model = Fixed_Mesh_2D(opt)
        opt['loss_type'] = 'mesh_loss'
        plot_initial_dataset_2d(dataset, opt)
