import sys
sys.path.append('../')
import time
import numpy as np
import torch
import torchquad
from torchquad import set_up_backend

from params_poisson import get_params, run_params, tf_sweep_args, set_seed, get_arg_list
from data_all import AllMeshInMemoryDataset
from utils_data import make_data_name
from utils_eval import plot_trained_dataset_1d, plot_trained_dataset_2d, evaluate_model_fine
from run_GNN import main as run_GNN, get_data
from firedrake_difFEM.difFEM_poisson_1d import Fixed_Mesh_1D, backFEM_1D
from firedrake_difFEM.difFEM_poisson_2d import Fixed_Mesh_2D, backFEM_2D
from utils_eval_Burgers import evaluate_model_fine_burgers, evaluate_model_fine_burgers_time_step, plot_trained_dataset_1d_burgers

def get_model(opt):
    mesh_dims = get_arg_list(opt['mesh_dims'])

    if opt['model'] in ['fixed_mesh_1D', 'backFEM_1D', 'fixed_mesh_2D', 'backFEM_2D']:
        if opt['model'] == 'fixed_mesh_1D':
            model = Fixed_Mesh_1D(opt)
        elif opt['model'] == 'backFEM_1D':
            model = backFEM_1D(opt)
        elif opt['model'] == 'fixed_mesh_2D':
            model = Fixed_Mesh_2D(opt)
        elif opt['model'] == 'backFEM_2D':
            model = backFEM_2D(opt)

        if 'all' in opt['data_name']:
            dataset = AllMeshInMemoryDataset(f"../data/{opt['data_name']}", mesh_dims, opt)
            mask = (dataset.data.pde_params['scale_value'] == opt['scale']) & (dataset.data.pde_params['mon_power'] == opt['mon_power'])
            test_dataset = dataset[mask]
        else:
            test_dataset = get_data(opt, train_test="test")

    else:
        model, train_dataset = run_GNN(opt)
        if 'all' in opt['data_name']:
            dataset = AllMeshInMemoryDataset(f"../data/{opt['data_name']}", mesh_dims, opt)
            mask = (dataset.data.pde_params['scale_value'] == opt['scale']) & (dataset.data.pde_params['mon_power'] == opt['mon_power'])
            test_dataset = dataset[mask]
        else:
            test_dataset = get_data(opt, train_test="test")

    return model, test_dataset

def main(opt):
    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed
    set_seed(opt['seed'])
    if torch.backends.mps.is_available():
        opt['device'] = torch.device('cpu') #mps
    elif torch.cuda.is_available():
        torchquad.enable_cuda(data_type='float32')
        set_up_backend("torch", data_type="float32")
        opt['device'] = torch.device('cuda')
    else:
        opt['device'] = torch.device('cpu')

    # get trained model and dataset
    opt = make_data_name(opt, train_test="train")

    start_train_time = time.time()
    print(f"Data name: {opt['data_name']}")
    model, dataset = get_model(opt)
    model.eval()
    end_train_time = time.time()

    start_eval_time = time.time()
    if opt['loss_type'] == 'modular':
        if 'burgers' in opt['grad_type']:
            if opt['num_eval_time_steps'] > 1:
                results_df_fine, times_df = evaluate_model_fine_burgers_time_step(model, dataset, opt)
            else:
                results_df_fine, times_df = evaluate_model_fine_burgers(model, dataset, opt)
    else:
        results_df_fine, times_df = evaluate_model_fine(model, dataset, opt, fine_eval=True)

    end_eval_time = time.time()

    dim = len(opt['mesh_dims'])
    if opt['show_dataset_plots']:
        show_mesh_evol_plots = opt['show_mesh_evol_plots']
        if dim == 1:
            if opt['loss_type'] == 'modular' and 'burgers' in opt['grad_type']:
                plot_trained_dataset_1d_burgers(dataset, model, opt, show_mesh_evol_plots=show_mesh_evol_plots)
            else:
                plot_trained_dataset_1d(dataset, model, opt, show_mesh_evol_plots=show_mesh_evol_plots)
        elif dim == 2:
            plot_trained_dataset_2d(dataset, model, opt, show_mesh_evol_plots=show_mesh_evol_plots)


if __name__ == "__main__":
    opt = get_params()
    opt = tf_sweep_args(opt)
    opt = run_params(opt)
    main(opt)
