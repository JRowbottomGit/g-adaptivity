import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from params_poisson import get_params, run_params, set_seed, get_arg_list
from utils_main import plot_training_evol
from data import generate_mesh_2d
from data import make_data_name, MeshInMemoryDataset
from data_mixed import MeshInMemoryDataset_Mixed
from data_mixed_loader import Mixed_DataLoader
from data_all import AllMeshInMemoryDataset
from GNN import GNN, MLP
from MON_NN import learn_Mon_RK4_1D, learn_Mon_RK4_2D
from firedrake_difFEM.difFEM_poisson_2d import gradient_meshpoints_2D
from firedrake_difFEM.difFEM_poisson_1d import gradient_meshpoints_1D


def get_data(opt, train_test="test"):
    mesh_dims = get_arg_list(opt['mesh_dims'])

    if opt['data_type'] == 'all':
        dataset = AllMeshInMemoryDataset(f"../data/{opt['data_name']}", mesh_dims, opt)
        mask = (dataset.data.pde_params['scale_value'] == opt['scale']) & (dataset.data.pde_params['mon_power'] == opt['mon_power'])
        dataset = dataset[mask]

    elif opt['data_type'] == 'randg_mix':
        dataset = MeshInMemoryDataset_Mixed(f"../data/{opt['data_name']}", "test", opt['num_train'], mesh_dims, opt)

    elif opt['dataset'] in ['fd_mmpde_1d', 'fd_mmpde_2d', 'fd_ma_2d', 'fd_M2N_2d']:  # firedrake MA grid with pde data
        num_data = opt['num_train'] if train_test == "train" else opt['num_test']
        dataset = MeshInMemoryDataset(f"../data/{opt['data_name']}", train_test, num_data, mesh_dims, opt)  # 11x11 node mesh

    elif opt['dataset'] == 'grid':
        dataset = generate_mesh_2d(mesh_dims[0], mesh_dims[1])

    if train_test == "train" and opt['train_frac'] is not None:
        sub_idxs = np.random.choice(len(dataset), int(len(dataset) * opt['train_frac']), replace=False)
        if opt['data_type'] == 'randg_mix':
            dataset.data_list = [dataset.data_list[i] for i in list(sub_idxs)]
        else:
            dataset = dataset[sub_idxs]
    elif train_test == "test" and opt['test_frac'] is not None:
        sub_idxs = np.random.choice(len(dataset), int(len(dataset) * opt['test_frac']), replace=False)
        if opt['data_type'] == 'randg_mix':
           dataset.data_list = [dataset.data_list[i] for i in list(sub_idxs)]
        else:
            dataset = dataset[sub_idxs]

    return dataset

def get_model(opt, dataset):
    if opt['model'] == 'GNN':
        model = GNN(dataset, opt).to(opt['device'])
    elif opt['model'] == 'MLP':
        model = MLP(dataset, opt).to(opt['device'])
    elif opt['model'] == 'learn_Mon_1D':
        model = learn_Mon_RK4_1D(dataset, opt).to(opt['device'])
    elif opt['model'] == 'learn_Mon_2D':
        model = learn_Mon_RK4_2D(dataset, opt).to(opt['device'])

    return model

def main(opt):
    opt = make_data_name(opt, "train")
    dataset = get_data(opt, train_test="train")
    if opt['data_type'] == 'randg_mix':
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=opt['batch_size'], shuffle=not(opt['overfit_num']),
                                  exclude_keys=exclude_keys, follow_batch=follow_batch, generator=torch.Generator(device=opt['device']))
    else:
        new_generator = torch.Generator(device=opt['device'])
        loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=not(opt['overfit_num']), generator=new_generator)

    model = get_model(opt, dataset)

    if opt['loss_type'] == 'mesh_loss':
        if opt['loss_fn'] == 'mse':
            loss_fn = F.mse_loss
        elif opt['loss_fn'] == 'l1':
            loss_fn = F.l1_loss
    elif opt['loss_type'] == 'pde_loss':
        loss_fn = F.mse_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['decay'])

    model.train()
    loss_list, mon_power_list, mon_scale_list,  mon_reg_list = [], [], [], []
    batch_loss_list = []
    best_loss = float('inf')

    for epoch in range(opt['epochs']):
        epoch_loss = 0
        model.epoch = epoch

        for i, data in enumerate(loader):
            data.idx = i
            optimizer.zero_grad()

            if opt['loss_type'] == 'mesh_loss':
                data = data.to(opt['device'])
                out = model(data)
                loss = loss_fn(out, data.x_phys)

            elif opt['loss_type'] == 'pde_loss':
                coeffs, x_phys, sol = model(data)
                loss = loss_fn(sol.to(opt['device']), data.u_true_fine_tensor.to(opt['device']))

            elif opt['loss_type'] == 'modular':
                # Get lenght of list
                if len(opt['mesh_dims'])==2:
                    x_phys = model(data)
                    x_phys_copy = x_phys.detach().clone()
                    loss, x_grads = gradient_meshpoints_2D(opt, data, x_phys_copy.detach())
                    pseudo_loss = sum(sum(x_phys * x_grads.detach()))
                elif len(opt['mesh_dims'])==1:
                    x_phys = model(data)[:,0]
                    x_phys_copy = x_phys.detach().clone()
                    loss, x_grads = gradient_meshpoints_1D(opt, data, x_phys_copy.detach())
                    pseudo_loss = sum(x_phys * x_grads.detach())

            epoch_loss += loss.item()
            if opt['loss_type'] == 'modular' and not opt['gnn_dont_train']:
                pseudo_loss.backward()
                optimizer.step()
            elif not opt['gnn_dont_train']:
                loss.backward()
                optimizer.step()

            print("     batch ", i, "batch loss: ", loss.item())

            batch_loss_list.append(loss.item())

        loss_list.append(epoch_loss)
        print("epoch: ", epoch, "epoch loss: ", epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dict = model.state_dict().copy()

    if opt['loss_type'] == 'modular':
        if 'save_model' in opt.keys() and opt['save_model'][0] == 'save':
            torch.save([loss_list,batch_loss_list], '../models/loss_list_' + opt['save_model'][1] + '.pth')

    #plot evolution of loss, mon_power and scale params
    if opt['show_train_evol_plots']:
        loss_fig = plot_training_evol(loss_list, "loss", batch_loss_list=batch_loss_list, batches_per_epoch=len(dataset)//opt['batch_size'])

    model.load_state_dict(best_dict)

    return model, dataset


if __name__ == "__main__":
    opt = get_params()
    opt = run_params(opt)
    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed
    main(opt)