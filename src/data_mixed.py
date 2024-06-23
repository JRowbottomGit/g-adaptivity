from torch_geometric.data import Batch
import sys
sys.path.append('../')

from data import *
from data_mixed_loader import Mixed_DataLoader


class MeshInMemoryDataset_Mixed(pyg.data.InMemoryDataset):
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

        super(MeshInMemoryDataset_Mixed, self).__init__(self.root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_list = torch.load(self.processed_paths[0])
        print("Dataset loaded with", len(self.data_list), "items.")

        custom_attributes_path = os.path.join(self.root, "processed", "custom_attributes.pt")
        if os.path.exists(custom_attributes_path):
            custom_attributes = torch.load(custom_attributes_path)
            self.mapping_tensor_fine = custom_attributes['mapping_tensor_fine']
            self.mapping_dict_fine = custom_attributes['mapping_dict_fine']
            self.orig_opt = custom_attributes['orig_opt']

        self.mesh = None
        self.mesh_deformed = None

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
            self.fine_mesh = UnitIntervalMesh(opt['eval_quad_points'] - 1, name="fine_mesh")
        elif self.dim == 2:
            self.fine_mesh = UnitSquareMesh(opt['eval_quad_points'] - 1, opt['eval_quad_points'] - 1, name="fine_mesh")

        if opt['data_type'] == 'randg_mix':
            self.mix_num_gauss = np.random.choice(opt['num_gauss_range'])

            if self.train_test == "train":
                min_mesh_points = opt['mesh_dims_train'][0][0]
                max_mesh_points = opt['mesh_dims_train'][-1][0]
            elif self.train_test == "test":
                min_mesh_points = opt['mesh_dims_test'][0][0]
                max_mesh_points = opt['mesh_dims_test'][-1][0]
            for n_meshpoints in range(min_mesh_points, max_mesh_points + 1):
                if self.dim == 1:
                    setattr(self, f"mesh_{n_meshpoints}", UnitIntervalMesh(n_meshpoints - 1, name=f"ref_mesh_{n_meshpoints}"))
                    setattr(self, f"mesh_deformed_{n_meshpoints}", UnitIntervalMesh(n_meshpoints - 1, name=f"deformed_mesh_{n_meshpoints}"))
                elif self.dim ==2:
                    setattr(self, f"mesh_{n_meshpoints}",
                            UnitSquareMesh(n_meshpoints - 1, n_meshpoints - 1, name=f"ref_mesh_{n_meshpoints}"))
                    setattr(self, f"mesh_deformed_{n_meshpoints}",
                            UnitSquareMesh(n_meshpoints - 1, n_meshpoints - 1, name=f"deformed_mesh_{n_meshpoints}"))
                with CheckpointFile(os.path.join(self.root, "processed", f"mesh_{n_meshpoints}.h5"), 'w') as mesh_file:
                    mesh_file.save_mesh(getattr(self, f"mesh_{n_meshpoints}"))
                with CheckpointFile(os.path.join(self.root, "processed", f"deformed_mesh_{n_meshpoints}.h5"), 'w') as mesh_file:
                    mesh_file.save_mesh(getattr(self, f"mesh_deformed_{n_meshpoints}"))

        # self.mapping_dict = mapping_dict
        # self.mapping_tensor = mapping_tensor
        # self.x_comp_shared = torch.tensor(self.mesh.coordinates.dat.data_ro)
        self.x_fine_shared = torch.tensor(self.fine_mesh.coordinates.dat.data_ro)

        if self.dim == 1:
            # mapping_dict, mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(self.x_comp_shared, n)
            mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, X_fd_vec_fine = map_firedrake_to_cannonical_ordering_1d(self.x_fine_shared, self.opt['eval_quad_points'])
        elif self.dim == 2:
            # mapping_dict, mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(self.x_comp_shared, n, m)
            mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, Y_fd_grid_fine, X_fd_vec_fine, Y_fd_vec_fine = map_firedrake_to_cannonical_ordering_2d(self.x_fine_shared, self.opt['eval_quad_points'], self.opt['eval_quad_points'])

        self.mapping_dict_fine = mapping_dict_fine
        self.mapping_tensor_fine = mapping_tensor_fine

        custom_attributes = {
            # 'x_comp_shared': self.x_comp_shared,
            # 'mapping_dict': self.mapping_dict,
            # 'mapping_tensor': self.mapping_tensor,
            'mapping_dict_fine': self.mapping_dict_fine,
            'mapping_tensor_fine': self.mapping_tensor_fine,
            'orig_opt': opt,
        }
        torch.save(custom_attributes, os.path.join(self.root, "processed", "custom_attributes.pt"))

        idx = 0
        data_list = []
        if opt['data_type'] in ['fixed', 'asym_sweep2d']:
            num_data_dict = {1: 9, 2: 25}
            self.num_data = num_data_dict[self.dim]

        for i in range(self.num_data):
            successful_eval = True

            if self.dim == 1:
                if opt['data_type'] == 'randg_mix':
                    if self.train_test == "train":
                        n = opt['mesh_dims_train'][np.random.choice(len(opt['mesh_dims_train']))][0]
                    elif self.train_test == "test":
                        n = opt['mesh_dims_test'][np.random.choice(len(opt['mesh_dims_test']))][0]
                # self.mesh = UnitIntervalMesh(n - 1, name="ref_mesh")
                # self.mesh_deformed = UnitIntervalMesh(n - 1, name="deformed_mesh")
                n_meshpoints = n
                self.mesh = getattr(self, f"mesh_{n_meshpoints}")
                self.mesh_deformed = getattr(self, f"mesh_deformed_{n_meshpoints}")

            elif self.dim == 2:
                if opt['data_type'] == 'randg_mix':
                    if self.train_test == "train":
                        n = opt['mesh_dims_train'][np.random.choice(len(opt['mesh_dims_train']))][0]
                        m = n
                        self.mix_num_gauss = np.random.choice(opt['num_gauss_range'])
                    elif self.train_test == "test":
                        n = opt['mesh_dims_test'][np.random.choice(len(opt['mesh_dims_test']))][0]
                        m = n
                        self.mix_num_gauss = np.random.choice(opt['num_gauss_range'])

                n_meshpoints = n
                self.mesh = getattr(self, f"mesh_{n_meshpoints}")
                self.mesh_deformed = getattr(self, f"mesh_deformed_{n_meshpoints}")

            data = firedrake_mesh_to_PyG(self.mesh)

            self.x_comp_shared = torch.tensor(self.mesh.coordinates.dat.data_ro)

            if self.dim == 1:
                data.mapping_dict, data.mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(self.x_comp_shared, n)
            elif self.dim == 2:
                data.mapping_dict, data.mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(self.x_comp_shared, n, m)

            c_list = []
            s_list = []
            if opt['data_type'] in ['randg']:
                for j in range(opt['num_gauss']):
                        c = np.random.uniform(0, 1, self.dim).astype('f') #float to match torch precison
                        s = np.random.uniform(0.1, 0.5, self.dim).astype('f')
                        c_list.append(c)
                        s_list.append(s)

            elif opt['data_type'] in ['randg_mix']:
                for j in range(self.mix_num_gauss):
                        c = np.random.uniform(0, 1, self.dim).astype('f') #float to match torch precison
                        s = np.random.uniform(0.1, np.sqrt(0.4), 1).astype('f').repeat(self.dim)
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

            elif opt['data_type'] == 'asym_sweep2d' and self.dim == 2:
                x_coord1 = i % 5 * 0.2 + 0.1
                y_coord1 = i // 5 * 0.2 + 0.1
                c1 = np.array([x_coord1, y_coord1])
                s1 = np.array([opt['scale_x'], opt['scale_y']])
                c_list.append(c1)
                s_list.append(s1)
                if opt['num_gauss'] == 2:
                    c2 = np.array([0.5, 0.5])
                    s2 = np.array([opt['scale_x'], opt['scale_y']])
                    c_list.append(c2)
                    s_list.append(s2)

            pde_params = {'centers': c_list, 'scales': s_list}
            pde_params['scale_list'] = s_list
            if opt['data_type'] not in ['randg', 'randg_mix']:
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

            # num_meshpoints = opt['mesh_dims'][0] if self.dim == 1 else opt['mesh_dims'][0]
            num_meshpoints = n if self.dim == 1 else n

            if self.dim == 1:
                eval_fct = poisson1d_fmultigauss_bcs
                #send in the deformed mesh so not to alter the original mesh
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
            if opt['data_type'] in ['randg', 'randg_mix']:
                filename_suffix = f"dim_{self.dim}" \
                                  f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data.h5"
            else:
                filename_suffix = f"dim_{self.dim}_scale_{round(data.pde_params['scale_value'], 2)}" \
                                  f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data.h5"

            pde_data_file = os.path.join(self.root, "processed", filename_suffix)

            with CheckpointFile(pde_data_file, 'w') as pde_file:
                # pde_file.save_mesh(self.mesh)
                # pde_file.save_mesh(self.mesh_deformed)
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
                data.uu_MA_tensor = torch.from_numpy(fcts_on_grids_dict['uu_ma'].dat.data)  # [self.mapping_tensor])
                data.u_true_MA_tensor = torch.from_numpy(fcts_on_grids_dict['u_true_ma'].dat.data)  # [self.mapping_tensor])
                data.f_MA_tensor = torch.from_numpy(fcts_on_grids_dict['f_ma'].dat.data)  # [self.mapping_tensor])
            else:
                data.uu_MA_tensor = torch.tensor([0.])
                data.u_true_MA_tensor = torch.tensor([0.])
                data.f_MA_tensor = torch.tensor([0.])

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            idx += 1

        # data, slices = self.collate(data_list)
        # data.apply(to_float32)
        # torch.save((data, slices), self.processed_paths[0])
        #above doesn't work for varying data sizes, so save the list of Data objects
        data_list = [data.apply(to_float32) for data in data_list]  # Apply any final processing to each data item
        torch.save(data_list, self.processed_paths[0])  # Save the list of Data objects

    def get(self, idx):
        # data = super().get(idx)
        data = self.data_list[idx]
        # data.x_comp = self.x_comp_shared.float()
        # data.x_comp = data.x_comp_shared.float()

        if isinstance(data.x_phys, np.ndarray):
            data.x_phys = torch.from_numpy(data.x_phys)

        mon_power = round(data.pde_params['mon_power'], 2)
        num_gauss = data.pde_params['num_gauss']
        if 'mon_reg' in data.pde_params:
            mon_reg = round(data.pde_params['mon_reg'], 2)
        if self.opt['data_type'] in ['randg', 'randg_mix']:
            filename_suffix = f"dim_{self.dim}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data.h5"
        else:
            # Retrieve scale and mon_power from data object
            scale = round(data.pde_params['scale_value'], 2)
            filename_suffix = f"dim_{self.dim}_scale_{scale}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data.h5"
        pde_data_file = os.path.join(self.root, "processed", filename_suffix)

        with CheckpointFile(pde_data_file, 'r') as pde_file:
            # data.mesh = pde_file.load_mesh("ref_mesh")
            # data.mesh_deformed = pde_file.load_mesh("deformed_mesh")
            n_meshpoints = int(np.sqrt(data.num_nodes))
            # Load the meshes
            with CheckpointFile(os.path.join(self.root, "processed", f"mesh_{n_meshpoints}.h5"), 'r') as mesh_file:
                mesh = mesh_file.load_mesh(f"ref_mesh_{n_meshpoints}")
            with CheckpointFile(os.path.join(self.root, "processed", f"deformed_mesh_{n_meshpoints}.h5"), 'r') as mesh_file:
                deformed_mesh = mesh_file.load_mesh(f"deformed_mesh_{n_meshpoints}")
            data.mesh = mesh
            data.mesh_deformed = deformed_mesh

            uu = pde_file.load_function(data.mesh, "uu")
            u_true = pde_file.load_function(data.mesh, "u_true")
            f = pde_file.load_function(data.mesh, "f")
            if data.successful_eval:
                uu_ma = pde_file.load_function(data.mesh_deformed, "uu_ma")
                u_true_ma = pde_file.load_function(data.mesh_deformed, "u_true_ma")
                f_ma = pde_file.load_function(data.mesh_deformed, "f_ma")
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

    @property
    def data(self):
        # This property will handle converting data_list into a batch when accessed
        return Batch.from_data_list(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Ensure the default implementation works, or override it correctly.
        # Using the base Dataset class method without modifications usually needs:
        return self.get(idx)

    def indices(self):
        # This should correctly reflect the indices for your dataset
        return list(range(self.__len__()))


def mixed_custom_collate(data_list, collate_keys=[]):
    collate_keys = ['ma_its', 'u_true_MA_tensor', 'f_fine_tensor', 'successful_eval',
    'uu_fine_tensor', 'eval_errors', 'u_true_fine_tensor',
    'corner_nodes', 'x_comp', 'x_phys',
    'f_MA_tensor', 'pde_params', 'uu_tensor', 'edge_index', 'u_true_tensor', 'uu_MA_tensor', 'f_tensor']

    batch_data = Batch()
    # Initialize containers for all attributes
    concat_attrs = {}  # Attributes that can be concatenated directly
    special_handling_attrs = {}  # Attributes that need special handling

    # Iterate over each data object
    for data in data_list:
        for key in data.keys():
            # Assuming `concat_attrs` is a dictionary of lists
            if key in collate_keys:  # Attributes that can be concatenated
                if key not in concat_attrs:
                    concat_attrs[key] = []
                concat_attrs[key].append(data[key])
            else:  # Special handling attributes
                if key not in special_handling_attrs:
                    special_handling_attrs[key] = []
                special_handling_attrs[key].append(data[key])

    # Concatenate concatenable attributes
    for key, values in concat_attrs.items():
        batch_data[key] = torch.cat(values, dim=0)

    # Handle non-concatenable attributes
    # For example, you could take the first one, average them, or create a list
    for key, values in special_handling_attrs.items():
        batch_data[key] = values  # or any other method you deem appropriate

    # Create batch indices for the concatenated data
    batch_data.batch = torch.tensor([i for i, data in enumerate(data_list) for _ in range(data.num_nodes)])

    return batch_data


def analyze_data_keys(data_list):
    all_keys = set()
    key_presence = {}

    # Gather all keys and initialize tracking for their presence across data items
    for data in data_list:
        for key in data.keys():
            all_keys.add(key)
            if key not in key_presence:
                key_presence[key] = [0] * len(data_list)

    # Mark presence of each key in each data item
    for i, data in enumerate(data_list):
        for key in data.keys():
            key_presence[key][i] = 1

    # Report results
    consistent_keys = {key for key, presences in key_presence.items() if all(presences)}
    inconsistent_keys = {key for key, presences in key_presence.items() if not all(presences)}

    print("Consistent keys across all data items:", consistent_keys)
    print("Inconsistent keys (missing in some data items):", inconsistent_keys)

    # Optionally, return this information for further processing
    return consistent_keys, inconsistent_keys


if __name__ == "__main__":
    opt = get_params()
    opt = run_params(opt)

    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed

    opt['data_type'] = 'randg_mix'
    opt['mesh_dims'] = [15, 15] #to make2d
    opt['mon_reg'] = 0.1
    opt['rand_gauss'] = True
    opt['num_train'] = 25#275#5#275#19#3#
    opt['num_test'] = 25#25#25#125#3#125#17#
    opt['mesh_dims_train'] = [[15, 15], [20, 20]] #[[3, 3], [4, 4]]#
    # opt['mesh_dims_test'] = [[12, 12]]#, [23, 23]]
    opt['mesh_dims_test'] = [[i, i] for i in range(12, 24, 1)]
    opt['num_gauss_range'] = [1, 2, 3, 5, 6]

    for train_test in ['test']:#'train', 'test']:
        opt = make_data_name(opt, train_test)
        if train_test == 'train':
            dataset = MeshInMemoryDataset_Mixed(f"../data/{opt['data_name']}", train_test, opt['num_train'], opt['mesh_dims'], opt)
        elif train_test == 'test':
            dataset = MeshInMemoryDataset_Mixed(f"../data/{opt['data_name']}", train_test, opt['num_test'], opt['mesh_dims'], opt)

        # Assuming you have a list of Data objects in `data_list`
        consistent_keys, inconsistent_keys = analyze_data_keys(dataset.data_list)

        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False,
                                  exclude_keys=exclude_keys, follow_batch=follow_batch, generator=torch.Generator(device=opt['device']))

        for i, data in enumerate(loader):
            print(f"Batch {i} retrieved successfully.")
            print(data)

        plot_initial_dataset_2d(dataset, opt)