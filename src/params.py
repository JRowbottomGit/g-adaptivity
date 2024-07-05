import argparse
import os
import numpy as np
import random
import torch


def run_params(opt):
    # Data params
    opt['pde_type'] = 'Poisson'  # 'Poisson', 'Burgers'

    if opt['pde_type'] == 'Burgers':
        data_dim = 1
        opt['data_type'] = "randg"
    else:
        data_dim = 2  #1 #2

    if data_dim == 1: #1D
        opt['data_type'] = "randg" #"all" #"randg"# 'randg_mix'
        opt['mesh_type'] = "mmpde"
        opt['dataset'] = f"fd_{opt['mesh_type']}_1d"
        opt['mesh_dims'] = [15]#10] #6]#11]21]51]101]
        opt['mon_reg'] = 0.1 #1.#0.10#0.01
        opt['num_gauss'] = 1

        #specify model
        opt['model'] = 'GNN'#'fixed_mesh_1D'#'backFEM_1D'#'GNN'

    elif data_dim == 2: #2D
        opt['data_type'] = "randg"  #'randg',#structured #"randg_mix" #"all"
        opt['mesh_type'] = "ma"  # ma or mmpde or M2N
        if opt['mesh_type'] == "M2N":
            opt['fast_M2N_monitor'] = "fast" #"slow" #"superslow" "fast"
            opt['M2N_alpha'] = 1.0
            opt['M2N_beta'] = 1.0
        opt['dataset'] = f"fd_{opt['mesh_type']}_2d"
        opt['mesh_dims'] = [11, 11] #[15, 15] #[11, 11]
        opt['mon_reg'] = 0.01 #.1 #0.1 #0.01

        #specify model
        opt['model'] = 'GNN' #'fixed_mesh_2D' 'backFEM_2D' #'GNN'

    #DATA args applies both 1/2D
    if opt['data_type'] == "all":
        opt['scale'] = 0.2 #0.1 #0.2 #0.3
        opt['mon_power'] = 0.2

    elif opt['data_type'] == 'structured':
        opt['num_gauss'] = 2
        opt['scale'] = 0.2
        opt['mon_power'] = 0.2

    elif opt['data_type'] == 'randg':
        opt['num_gauss'] = 2
        opt['rand_gauss'] = True
        opt['num_train'] = 25
        opt['num_test'] = 25

    elif opt['data_type'] == 'randg_mix':
        opt['rand_gauss'] = True
        opt['num_train'] = 275 #25#5#275
        opt['num_test'] = 125 #25#3#125
        opt['mesh_dims_train'] = [[15, 15], [20, 20]]
        opt['mesh_dims_test'] = [[i, i] for i in range(12, 24, 1)]
        opt['num_gauss_range'] = [1,2,3,5,6]

    opt['fix_boundary'] = True #False
    opt['eval_quad_points'] = 101
    opt['stiff_quad_points'] = 3
    opt['load_quad_points'] = 101

    #MODEL args
    if opt['model'] == 'fixed_mesh_1D':
        opt['solver'] = 'firedrake'#'torch_FEM' # 'firedrake'
        opt['evaler'] = 'analytical' #'fd_fine' #'fd_coarse' #'analytic'
        opt['loss_type'] == 'mesh_loss' #'pde_loss'

    elif opt['model'] == 'fixed_mesh_2D':
        opt['solver'] = 'firedrake'
        opt['evaler'] = 'analytical' #'fd_fine' #'fd_coarse' #'analytic'
        opt['loss_type'] = 'mesh_loss' #'pde_loss'

    elif opt['model'] == 'backFEM_1D':
        opt['loss_type'] = 'pde_loss'  # 'mesh_loss' #'pde_loss' #,'pinn_loss'
        opt['solver'] = 'torch_FEM'
        opt['evaler'] = 'analytical' #'fd_fine' #'fd_coarse' #'analytic'
        opt['mesh_params'] = "internal"# "internal" # "all" #grad wrt to all mesh params or just internal

        opt['epochs'] = 10 #20000 #10 50 500  #note this has a very big factor on performance!
        #note the more nodes the smaller the lr needs to be to stop crossing
        if opt['mesh_dims'][0] == 11:
            opt['lr'] = 0.05
        elif opt['mesh_dims'][0] == 21:
            opt['lr'] = 0.01
        elif opt['mesh_dims'][0] == 51:
            opt['lr'] = 0.001

    elif opt['model'] == 'backFEM_2D':
        opt['loss_type'] = 'pde_loss'
        opt['evaler'] = 'analytical'
        opt['solver'] = 'torch_FEM'
        opt['epochs'] = 200
        opt['lr'] = 0.2 #5#0.05
        opt['load_quad_points'] = 101

    elif opt['model'] == 'GNN':
        opt['epochs'] = 1#0
        opt['gnn_dont_train'] = False #True
        opt['loss_type'] = 'pde_loss' #'pde_loss'  # 'mesh_loss' #'modular' #,'pinn_loss'
        opt['loss_fn'] = 'l1'
        opt['solver'] = 'torch_FEM'

        #features
        opt['gnn_inc_feat_f'] = True #False #whether to include pde features in GNN
        opt['gnn_inc_feat_uu'] = True #True #False #whether to include uu features in GNN
        opt['gnn_inc_glob_feat_f'] = False #whether to include global features in GNN
        opt['gnn_inc_glob_feat_uu'] = False #False #True
        opt['gnn_normalize'] = False #True #False #whether to normalise the raw features

        #model
        opt['conv_type'] = 'GRAND_plus'#'GAT_plus' #'GRAND_plus #'GRAND'#'GCN'#'GAT'#'TRANS'
        opt['gat_plus_type'] = 'GAT_res_lap' #'GAT_res_lap' #'GAT_lin'
        encdec = 'identity'
        opt['enc'] = encdec
        opt['dec'] = encdec

        opt['residual'] = True
        opt['share_conv'] = True  #True #False
        opt['non_lin'] = 'identity'
        opt['num_layers'] = 4  # 2
        opt['time_step'] = 0.1 #0.05#0.1 #1.
        opt['hidden_dim'] = 8 #16#32
        opt['global_feat_dim'] = 8
        opt['lr'] = 0.001

    #overwrite params for Burgers example
    if opt['pde_type'] == 'Burgers':
        opt['gauss_amplitude'] = 0.25
        opt['burgers_limits'] = 3.0
        opt['num_train'] = 20
        opt['num_test'] = 5
        opt['scale'] = 0.1
        opt['mon_reg'] = 0.1
        opt['num_gauss'] = 1
        opt['loss_type'] = 'modular'
        if opt['loss_type'] == 'modular':
            opt['mesh_dims'] = [21]
            opt['conv_type'] = 'GRAND'
            opt['loss_type'] = 'modular'
            opt['grad_type'] = 'burgers_timestep_loss_direct_mse'
            opt['epochs'] = 100
            opt['global_feat_dim'] = 8
            if 'burgers' in opt['grad_type']:
                opt['num_fine_mesh_points'] = 40
                opt['gnn_inc_feat_f'] = False
                opt['tau'] = 1 / 20.0
                opt['nu'] = 0.001
                opt['num_time_steps'] = 1
                opt['num_eval_time_steps'] = 20

    return opt


def t_or_f(tf_str):
    if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
        return True
    elif tf_str == "False" or tf_str == "false" or (type(tf_str) == bool and not tf_str):
        return False
    else:
        return tf_str

def tf_sweep_args(opt):
    for arg in list(opt.keys()):
        str_tf = opt[arg]
        bool_tf = t_or_f(str_tf)
        opt[arg] = bool_tf
    return opt

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_arg_list(arg_list):
    print(arg_list, len(arg_list), type(arg_list))
    if type(arg_list[0]) == int:
        pass
    else:
        arg_list = eval(arg_list[0]) #fix to deal with list of args input
    return arg_list


def get_params():
    parser = argparse.ArgumentParser()
    #data params
    parser.add_argument('--dataset', type=str, default='grid', choices=['fd_mmpde_1d', 'fd_mmpde_2d','fd_ma_2d','grid', 'noisey_grid','triangles'], help="high level data type")
    parser.add_argument('--data_type', type=str, default='randg', choices=['all', 'structured', 'randg', 'randg_mix'], help="data desriptor")
    parser.add_argument('--fast_M2N_monitor', type=str, default="slow", help="fast M2N monitor", choices=['fast', 'slow', 'superslow'])
    parser.add_argument('--M2N_alpha', type=float, default=None, help="M2N alpha")
    parser.add_argument('--M2N_beta', type=float, default=None, help="M2N beta")
    parser.add_argument('--mesh_type', type=str, default='ma', help="mesh type", choices=['mmpde', 'ma', 'M2N'])
    parser.add_argument('--data_name', type=str, default='test', help="data path desriptor")
    parser.add_argument('--data_train_test', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_train', type=int, default=100, help="number of training data points")
    parser.add_argument('--num_test', type=int, default=25, help="number of test data points")

    #mix data params
    parser.add_argument('--mesh_dims_train', nargs='+', default=[[15, 15], [20, 20]], help='dimensions of mesh - width, height')
    parser.add_argument('--mesh_dims_test', nargs='+', default=[[i, i] for i in range(12, 24, 1)], help='dimensions of mesh - width, height')
    parser.add_argument('--num_gauss_range', nargs='+', default=[1, 2, 3, 5, 6], help='number of Gaussians in u')
    parser.add_argument('--train_frac', type=float, default=None, help="fraction of training data")
    parser.add_argument('--test_frac', type=float, default=None, help="fraction of test data")

    #mesh params
    parser.add_argument('--mesh_dims', nargs='+', default=[10, 10], help='dimensions of mesh - width, height')
    parser.add_argument('--fix_boundary', type=str, default="True", help="fix boundary nodes")
    parser.add_argument('--mon_reg', type=float, default=0.1, help="regularisation term in MMPDE5")
    parser.add_argument('--mon_power', type=float, default=0.2, help="power term in MMPDE5")
    # parser.add_argument('--tau', type=float, default=0.006, help="speed parameter in MMPDE5")

    #pde params
    parser.add_argument('--pde_type', type=str, default='Poisson', choices=['Poisson', 'Burgers'], help="PDE type")
    parser.add_argument('--boundary', type=str, default='dirichlet')
    parser.add_argument('--num_gauss', type=int, default=1, help='number of Gaussians in u')
    parser.add_argument('--rand_gauss', type=bool, default=False, help='whether Gaussians in u random c/s')
    parser.add_argument('--scale', type=float, default=0.2, help="variance of Gaussian solution u")
    parser.add_argument('--center', type=float, default=0.5, help="center of Gaussian solution u")

    #fem params
    parser.add_argument('--eval_quad_points', type=int, default=101, help='number of quad points')
    parser.add_argument('--stiff_quad_points', type=int, default=3, help='number of quad points per interval')
    parser.add_argument('--load_quad_points', type=int, default=101, help='number of quad points per interval')

    #model params
    parser.add_argument('--model', type=str, default='GNN', choices=['fixed_mesh_1D','fixed_mesh_2D','backFEM_1D','backFEM_2D','GNN','MLP'])

    #shared params
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--global_feat_dim', type=int, default=8)
    parser.add_argument('--enc', type=str, default='identity', choices=['identity', 'lin_layer', 'mlp'])
    parser.add_argument('--dec', type=str, default='identity', choices=['identity', 'lin_layer', 'mlp'])
    parser.add_argument('--non_lin', type=str, default='identity', choices=['identity', 'relu', 'tanh', 'sigmoid', 'leaky_relu'])
    parser.add_argument('--residual', type=str, default="True")
    parser.add_argument('--mesh_params', type=str, default="internal", help="internal, all")
    parser.add_argument('--time_step', type=float, default=0.1, help="time_step")

    #GNN
    parser.add_argument('--conv_type', type=str, default='GCN', choices=['GCN', 'GAT', 'GRAND', 'GRAND_plus', 'GAT_plus', 'Laplacian'])
    parser.add_argument('--share_conv', type=str, default="True")
    parser.add_argument('--gnn_inc_feat_f', type=str, default="True", help="include pde features")
    parser.add_argument('--gnn_inc_feat_uu', type=str, default="False", help="include u,u features")
    parser.add_argument('--gnn_inc_glob_feat_f', type=str, default="True", help="include global features")
    parser.add_argument('--gnn_inc_glob_feat_uu', type=str, default="True", help="include global uu features")
    parser.add_argument('--gnn_normalize', type=str, default="False", help="normalise the raw features")

    #GNN regularisation params
    parser.add_argument('--self_loops', type=str, default="False", help="include self loops.")
    parser.add_argument('--softmax_temp_type', type=str, default=None, choices=['none','fixed','learnable'], help="use fixed or variable softmax temperature")
    parser.add_argument('--softmax_temp', type=float, default=2.0, help="Fixed softmax temperature value.")
    parser.add_argument('--learn_step', type=str, default="False", help="learn the step size.")
    parser.add_argument('--gnn_dont_train', type=str, default="False", help="gnn_dont_train.")
    parser.add_argument('--reg_skew', type=str, default="False", help="reg_skew with triangle area weighted attention")

    #GAT params
    parser.add_argument('--gat_plus_type', type=str, default='GAT_res_lap', choices=['GAT_res_lap', 'GAT_lin', 'GAT', 'GAT_phys', 'None'])

    #Burger's params
    parser.add_argument('--gauss_amplitude', type=float, default=0.25, help="amplitude of Gaussians")
    parser.add_argument('--burgers_limits', type=float, default=3.0, help="spatial domain limits")
    parser.add_argument('--plots_multistep_eval', type=str, default="False")
    parser.add_argument('--plots_mesh_movement', type=str, default="False")

    #training params
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--overfit_num', nargs='+', default=None, help="list of data points to overfit to")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--decay', type=float, default=0.)
    parser.add_argument('--loss_type', type=str, default='mesh_loss', choices=['mesh_loss','pde_loss','pinn_loss'])
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['mse', 'l1'])
    parser.add_argument('--solver', type=str, default='torch_FEM', choices=['firedrake', 'torch_FEM', 'BVP'])
    parser.add_argument('--evaler', type=str, default='analytical', choices=['fd_fine', 'fd_coarse', 'analytical'])

    #plot params
    parser.add_argument('--show_plots', type=str, default="True", help="flag to show plots")
    parser.add_argument('--show_dataset_plots', type=str, default="True", help="flag to show full test dataset plots")
    parser.add_argument('--show_train_evol_plots', type=str, default="True", help="flag to show evolution of training")
    parser.add_argument('--show_mesh_evol_plots', type=str, default="True", help="flag to show evolution of mesh plots")
    parser.add_argument('--show_mesh_plots', type=str, default="False", help="flag to show individual mesh plots")

    args = parser.parse_args()
    args = vars(args)
    return args