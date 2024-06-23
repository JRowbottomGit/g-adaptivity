import sys
import time
from deepdiff import DeepDiff

sys.path.append('../')
import os
import numpy as np
import torch
import torchquad
from torchquad import set_up_backend #https://torchquad.readthedocs.io/en/main/tutorial.html#imports
import wandb
from utils_main import plot_training_evol
from torch.utils.tensorboard import SummaryWriter

from params_burgers import get_params, run_params, tf_sweep_args, cond_sweep_args, set_seed, get_arg_list
from data import plot_initial_dataset_2d
from data_all import AllMeshInMemoryDataset
from utils_data import make_data_name
from utils_main import vizualise_grid_with_edges
from utils_eval import update_mesh_coords, plot_trained_dataset_1d, plot_trained_dataset_2d, evaluate_model_fine
from run_GNN import main as run_GNN, get_data
from firedrake_difFEM.solve_poisson import poisson2d_fgauss_b0, poisson1d_fmultigauss_bcs, poisson2d_fmultigauss_bcs, plot_solutions
from firedrake_difFEM.difFEM_poisson_1d import Fixed_Mesh_1D, backFEM_1D
from firedrake_difFEM.difFEM_poisson_2d import Fixed_Mesh_2D, backFEM_2D
from utils_eval_Burgers import evaluate_model_fine_burgers, evaluate_model_fine_burgers_time_step, plot_trained_dataset_1d_burgers


def compare_dicts_with_exceptions(dict1, dict2, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = []

    keys1 = set(dict1.keys()) - set(ignore_keys)
    keys2 = set(dict2.keys()) - set(ignore_keys)

    common_keys = keys1 & keys2
    added_keys = keys2 - keys1
    removed_keys = keys1 - keys2

    modified_keys = {key for key in common_keys if dict1[key] != dict2[key]}

    return modified_keys

def get_model(opt, tb_writer=None):
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
            #work around to stop circular import
            dataset = AllMeshInMemoryDataset(f"../data/{opt['data_name']}", mesh_dims, opt)
            mask = (dataset.data.pde_params['scale_value'] == opt['scale']) & (dataset.data.pde_params['mon_power'] == opt['mon_power'])
            test_dataset = dataset[mask]
        else:
            test_dataset = get_data(opt, train_test="test")

    else:
        model, train_dataset = run_GNN(opt)
        #torch.save(model.state_dict(), "../output/model_end.pt")
        #model.load_state_dict(torch.load("../output/model_end.pt"))
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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'mps'
    # opt['device'] = device
    if torch.backends.mps.is_available():
        opt['device'] = torch.device('cpu')

        # print("MPS Fallback:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # print("MPS Fallback:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        # opt['device'] = torch.device('mps')
        # NotImplementedError: The operator 'aten::scatter_reduce.two_out' is not currently implemented for the MPS device.
        # If you want this op to be added in priority during the prototype phase of this feature,
        # please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix,
        # you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
        # export PYTORCH_ENABLE_MPS_FALLBACK = 1
        #echo $PYTORCH_ENABLE_MPS_FALLBACK

    elif torch.cuda.is_available():
        torchquad.enable_cuda(data_type='float32')
        set_up_backend("torch", data_type="float32")
        opt['device'] = torch.device('cuda')
    else:
        opt['device'] = torch.device('cpu')

    # get trained model and dataset
    opt = make_data_name(opt, train_test="train")


    if opt['wandb']:
        if opt['wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"

        if 'wandb_run_name' in opt.keys():
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                                   name=opt['wandb_run_name'], reinit=True, config=opt, allow_val_change=True)#, sync_tensorboard=opt['tensorboard'])
        else:
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                                   reinit=True, config=opt, allow_val_change=True)#, sync_tensorboard=opt['tensorboard'])
        if opt['tensorboard']:
            wandb.tensorboard.patch(save=True, tensorboard_x=False, pytorch=True)
        opt = wandb.config  # access all HPs through wandb.config, so logging matches execution!

        if opt['wandb_sweep']:
            opt = cond_sweep_args(opt)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    if opt['tensorboard']:
        tb_writer = SummaryWriter()
    else:
        tb_writer = None


    start_train_time = time.time()
    print(f"Data name: {opt['data_name']}")
    # Manual saving of model
    if 'save_model' in opt.keys() and opt['save_model'][0]=='save':
        model, dataset = get_model(opt, tb_writer=tb_writer)
        torch.save(model.state_dict(), '../models/model_'+opt['save_model'][1]+'.pth')
        torch.save(opt, '../models/parameters_'+opt['save_model'][1]+'.pth')
    elif 'save_model' in opt.keys() and opt['save_model'][0] == 'load':
        opt_saved=torch.load('../models/parameters_'+opt['save_model'][1]+'.pth')
        compare_dicts_with_exceptions(opt, opt_saved, ignore_keys=['save_model', 'seed', 'num_eval_time_steps'])
        if len(compare_dicts_with_exceptions(opt, opt_saved, ignore_keys=['save_model', 'seed'])) > 0:
            diff = DeepDiff(opt, opt_saved, ignore_order=True)
            print(diff)
            input('Parameters of saved model do not match current parameters...if you still want to continue hit Enter')
            # return
        epochs_val_save=opt['epochs']
        opt['epochs'] = 1
        model, dataset = get_model(opt, tb_writer=tb_writer)
        model.load_state_dict(torch.load('../models/model_'+opt['save_model'][1]+'.pth'))
        opt['epochs'] = epochs_val_save
    else:
        model, dataset = get_model(opt, tb_writer=tb_writer)

    model.eval()
    end_train_time = time.time()

    start_eval_time = time.time()
    # # results_df = evaluate_model(model, dataset, opt) #loop through the dataset and solve the pde using FEM for each mesh
    # #todo: add option to evaluate on coarse mesh
    # # results_df_fine_coarse = evaluate_model_fine(model, dataset, opt, fine_eval=False)
    if opt['loss_type'] == 'modular':
        if 'burgers' in opt['grad_type']:
            if opt['num_eval_time_steps'] > 1:
                results_df_fine, times_df = evaluate_model_fine_burgers_time_step(model, dataset, opt)
            else:
                results_df_fine, times_df = evaluate_model_fine_burgers(model, dataset, opt)
            #todo GM
    else:
        results_df_fine, times_df = evaluate_model_fine(model, dataset, opt, fine_eval=True)


    end_eval_time = time.time()

    dim = len(opt['mesh_dims'])
    # show full dataset plots
    if opt['show_dataset_plots']:
        show_mesh_evol_plots = False if opt['wandb_sweep'] else opt['show_mesh_evol_plots']
        if dim == 1:
            if opt['loss_type'] == 'modular' and 'burgers' in opt['grad_type']:
                plot_trained_dataset_1d_burgers(dataset, model, opt, show_mesh_evol_plots=show_mesh_evol_plots)
            else:
                plot_trained_dataset_1d(dataset, model, opt, show_mesh_evol_plots=show_mesh_evol_plots)
        elif dim == 2:
            plot_trained_dataset_2d(dataset, model, opt, show_mesh_evol_plots=show_mesh_evol_plots)

    # #show individual mesh plots
    # if opt['show_mesh_plots']:
    #     plot_individual_meshes(dataset, model, opt)
    #
    # if opt['show_monitor_plots'] and opt['model'] == 'GNN':
    #     #model with init state dict
    #     # eval_learned_monitor(model, dataset, opt, show_mesh_evol_plots=True)
    #     eval_learned_monitor(model, dataset, opt, show_mesh_evol_plots=True)

    if dim == 2 and opt['plot_fig1']:
        # plot_fig1_2d(dataset, model, opt, plot_type="err")
        # plot_fig1_2d(dataset, model, opt, plot_type="u_true")
        # plot_fig1_2d(dataset, model, opt, plot_type="uu")
        for i in range(2):
            plot_fig1_2d(dataset, model, opt, plot_type="err", item=i, save_plots=True)

    # plot_fig1_2d(dataset, model, opt, solver='torch_FEM')

    if opt['wandb']:
        # results_table = wandb.Table(dataframe=results_df)
        results_table_fine = wandb.Table(dataframe=results_df_fine.round(4))
        results_table_fine_describe = wandb.Table(dataframe=results_df_fine.describe().reset_index().round(4))
        times_table = wandb.Table(dataframe=times_df.round(6))
        times_table_describe = wandb.Table(dataframe=times_df.describe().reset_index().round(6))

        wandb.log({#"results_table": results_table,
                    "results_table_fine": results_table_fine,
                    "results_table_fine_describe": results_table_fine_describe,
                    "times_table": times_table,
                    "times_table_describe": times_table_describe})

        #average results over rows, convert to dict using only the suffix of the column name
        # results_dict = {f"{key.split('.')[-1]}_coarse": results_df.mean()[key] for key in results_df.columns}
        results_dict_fine = {f"{key.split('.')[-1]}_fine": results_df_fine.mean()[key] for key in results_df_fine.columns} #for backwards compatibility
        times_dict = {f"{key.split('.')[-1]}": times_df.mean()[key] for key in times_df.columns}
        headline_results = {#**results_dict,
                            **results_dict_fine,
                            **times_dict,
                       "train_time": end_train_time - start_train_time,
                       "eval_time": end_eval_time - start_eval_time}
        wandb.log(headline_results)

    # if opt['wandb_log_plots']:
    #     #save plots as wandb Image type
    #     wandb.log({"original_mesh": wandb.Image(orig_mesh),
    #                 "noisey_mesh": wandb.Image(noisey_mesh),
    #                 "learned_mesh": wandb.Image(learned_mesh)})

    if opt['wandb']:
        wandb.finish()


if __name__ == "__main__":
    opt = get_params()
    opt = tf_sweep_args(opt)
    if not opt['wandb_sweep']:
        opt = run_params(opt)

    #@GM moved all this to bottom of params file as breaks pipeline for sweeps search if opt['loss_type'] == 'modular'
    #opt['epochs']=20
    opt['conv_type'] = 'GRAND'
    #opt['conv_type'] = 'GRAND'
    #opt['loss_type'] = 'modular'
    # opt['grad_type'] = 'Burgers_timestep_loss_direct_mse'
    opt['reg_skew']=False
    opt['epochs']=50
    opt['num_eval_quad_points'] = 200
    opt['num_fine_mesh_points'] = 40
    opt['time_step'] = 0.1
    opt['num_layers'] = 4
    opt['gauss_amplitude'] = 0.25
    opt['load_quad_points'] = 101
    opt['num_train'] = 20 # 275
    opt['num_test'] = 1#125  # 125
    opt['scale'] = 0.1
    opt['data_burgers'] = True
    opt['plots_multistep_eval'] = False#True#False#True
    opt['plots_mesh_movement'] = True
    opt['mon_reg']=0.1
    opt['num_gauss'] = 1
    opt['burgers_limits'] = 3.0
    #
    # opt['tau'] = 0.01
    # opt['nu'] = 0.001

    #opt['batch_size'] = 25
    # opt['num_samples'] = 5 #I think this was used in conditional sweep
    main(opt)


    loss_list,batch_loss_list=torch.load('../models/loss_list_' + opt['save_model'][1] + '.pth')

    loss_fig = plot_training_evol(loss_list, "loss",plot_show=True, batch_loss_list=batch_loss_list,
                                  batches_per_epoch=opt['num_train'] // opt['batch_size'])
