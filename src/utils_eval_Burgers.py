import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.interpolate import UnivariateSpline

from firedrake_difFEM.difFEM_1d import fn_expansion,gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse,plotting_data_burgers,get_Burgers_initial_coeffs,get_Burgers_initial_coeffs_old,torch_FEM_Burgers_1D,u_true_exact_1d_vec,remesh_1d
from classical_meshing.ma_mesh_1d import MMPDE5_1d_burgers,diag_hessian
from utils_eval import *


def evaluate_model_fine_burgers(model, dataset, opt):
    dim = len(dataset.opt['mesh_dims'])
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

        if dim == 1:
            num_meshpoints = dataset.opt['mesh_dims'][0]

        data.idx = i

        # Regular grid solution
        grid_coords = torch.linspace(0, 1, num_meshpoints)

        L2_grid, _ = gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt, data, grid_coords)

        #2) MMPDE5 Mesh
        deformed_mesh = dataset.mesh_deformed
        deformed_mesh_points=torch.tensor(deformed_mesh.coordinates.dat.data_ro)

        L2_MA, _ = gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt, data, deformed_mesh_points)

        #3) Get the model deformed mesh from trained model
        try:
            MA_time = data.build_time.item()
        except:
            MA_time = 0.0

        # Start the timer
        start_time = time.process_time()

        MLmodel_coords = model(data)

        # End the timer
        MLmodel_time = time.process_time()-start_time

        MLmodel_coords = MLmodel_coords.squeeze()

        #evaluate error
        L2_MLmodel,_ = gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt, data, MLmodel_coords)

        # Calculate error reduction ratios
        L2_reduction_MA = calculate_error_reduction(L2_grid, L2_MA)
        L2_reduction_MLmodel = calculate_error_reduction(L2_grid, L2_MLmodel)

        results.append({
            'L2_grid': L2_grid.item(),
            'L2_MA': L2_MA.item(),
            'L2_MLmodel': L2_MLmodel.item(),
            'L2_reduction_MA': L2_reduction_MA.item(),
            'L2_reduction_MLmodel': L2_reduction_MLmodel.item()
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

def evaluate_model_fine_burgers_time_step(model, dataset, opt):
    dim = len(dataset.opt['mesh_dims'])

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
            c_list = data.batch_dict[0]['pde_params']['centers']
            s_list = data.batch_dict[0]['pde_params']['scales']
        else:
            c_list = data.pde_params['centers'][0]
            s_list = data.pde_params['scales'][0]

        if dim == 1:
            num_meshpoints = dataset.opt['mesh_dims'][0]

        data.idx = i

        # Set up mesh
        num_fine_mesh_points = opt['num_fine_mesh_points']
        fine_mesh_points = torch.linspace(0, 1, num_fine_mesh_points)
        load_quad_points = opt['load_quad_points']

        c_list = [torch.tensor(c) for c in data.pde_params['centers'][0]]
        s_list = [torch.tensor(s) for s in data.pde_params['scales'][0]]
        u0 = lambda x: opt['gauss_amplitude'] * u_true_exact_1d_vec(x, c_list, s_list)

        # Regular grid solution
        grid_coords = torch.linspace(0, 1, num_meshpoints)

        quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

        u0_coeffs, u0_coeffs_fine = get_Burgers_initial_coeffs(fine_mesh_points, num_fine_mesh_points, grid_coords,
                                                               num_meshpoints, u0, load_quad_points, opt)
        un_coeffs = u0_coeffs.clone().detach()
        un_coeffs_fine = u0_coeffs_fine

        sol_fine=torch.zeros(opt['num_eval_time_steps'],opt['num_time_steps'],opt['eval_quad_points'])

        with torch.no_grad():
            for l in range(opt['num_eval_time_steps'] - 1):
                print(f"Time step {l}")
                print(f"Results length: {len(results)}")
                grid_coords_old = grid_coords.clone().detach()
                for j in range(opt['num_time_steps']):
                    un_coeffs, _, sol_grid, _, _ = torch_FEM_Burgers_1D(opt, grid_coords, quad_points,
                                                                   num_meshpoints, un_coeffs.clone().detach())
                    #sol_fine_old[l, j, :] = sol_temp
                    un_coeffs_fine, _, sol_temp, _, _ = torch_FEM_Burgers_1D(opt, fine_mesh_points, quad_points, num_fine_mesh_points, un_coeffs_fine.clone().detach())
                    sol_fine[l, j, :] = sol_temp

                    if opt['plots_multistep_eval'] and l % 1 == 0:
                        plt.figure()
                        plt.plot(quad_points.detach().numpy(), sol_grid.detach().numpy())
                        plt.plot(quad_points.detach().numpy(), sol_fine[l,j,:].detach().numpy())
                        plt.scatter(grid_coords, 0.0 * grid_coords, label='Mesh points', color='blue', marker='o')
                        plt.xlabel('x')
                        plt.ylabel('u')
                        plt.title(f"Evolution of the Solution)")
                        plt.show()

                        L2_grid = F.mse_loss(sol_grid, sol_fine[l, j, :])
                        print(L2_grid)
                        input('Press Enter1 to continue...')

        L2_grid = F.mse_loss(sol_grid,sol_fine[l,j,:])

        #2) MMPDE5 Mesh
        deformed_mesh = dataset.mesh_deformed
        MA_meshpoints=torch.tensor(deformed_mesh.coordinates.dat.data_ro, dtype=torch.float32)

        start_time = time.process_time()
        u0_coeffs_MA, _ = get_Burgers_initial_coeffs(fine_mesh_points, num_fine_mesh_points, MA_meshpoints,
                                                               num_meshpoints, u0, load_quad_points, opt)
        un_coeffs_MA = u0_coeffs_MA.clone().detach()
        mesh_time_MA=0.0
        if opt['plots_mesh_movement']:
            MA_mesh_save=torch.zeros(opt['num_eval_time_steps']*opt['num_time_steps'],num_meshpoints)
            timesteps=torch.zeros(opt['num_eval_time_steps']*opt['num_time_steps'])
            k=0

        if opt['plots_multistep_eval']:
            plt.plot()
            plt.figure()
            plt.plot(MA_meshpoints.detach().numpy(), u0_coeffs_MA.detach().numpy())
            plt.plot(fine_mesh_points.detach().numpy(), u0_coeffs_fine.detach().numpy())
            plt.scatter(MA_meshpoints, 0.0 * MA_meshpoints, label='Mesh points', color='blue', marker='o')
            plt.show()
            input('Press Enter2 to continue...')
        with torch.no_grad():
            for l in range(opt['num_eval_time_steps']-1):
                print(f"Time step {l}")
                MA_meshpoints_old = MA_meshpoints.clone().detach()
                for j in range(opt['num_time_steps']):
                    un_coeffs_MA, _, sol_MA, _, _=torch_FEM_Burgers_1D(opt, MA_meshpoints, quad_points, num_meshpoints, un_coeffs_MA.clone().detach())


                    if opt['plots_multistep_eval'] and l % 1 == 0:
                        plt.plot()
                        plt.figure()
                        plt.plot(quad_points.detach().numpy(), sol_MA.detach().numpy())
                        #plt.plot(MA_meshpoints.detach().numpy(), un_coeffs.detach().numpy())
                        plt.plot(quad_points.detach().numpy(), sol_fine[l,j,:].detach().numpy())
                        plt.scatter(MA_meshpoints, 0.0 * MA_meshpoints, label='Mesh points', color='blue', marker='o')
                        plt.show()

                        L2_MA = F.mse_loss(sol_MA, sol_fine[l, j, :])
                        print(L2_MA, L2_grid)

                        input('Press Enter2 to continue...')


                # Deform MMPDE5 meshpoints
                # Create a spline interpolant
                #spline = UnivariateSpline(MA_meshpoints.detach().numpy(), un_coeffs.detach().numpy(), s=0)  # s=0 ensures it interpolates through the data points
                spline = UnivariateSpline(quad_points.detach().numpy(), sol_fine[l,j,:].detach().numpy(), s=0)  # s=0 ensures it interpolates through the data points

                # Differentiate the spline to get the first and second derivatives
                #spline_first_derivative = spline.derivative(n=1)
                spline_second_derivative = spline.derivative(n=2)
                max_value=spline_second_derivative(fine_mesh_points.detach().numpy()).max()

                m = lambda x: torch.tensor((opt['mon_reg'] + (spline_second_derivative(x.detach().numpy())/max_value)**2.0) ** opt['mon_power'])
                #m = lambda x: torch.tensor((opt['mon_reg']**2 + (spline_first_derivative(x.detach().numpy())) ** 2.0) **0.5)#** opt['mon_power'])

                start_time2 = time.process_time()
                MA_meshpoints,numMMPDEsteps,_ =MMPDE5_1d_burgers(m, MA_meshpoints.clone().detach(), num_meshpoints)
                mesh_time_MA += time.process_time()-start_time2
                #print('Mesh time:', mesh_time_MA)
                #print('increment in time:', time.process_time()-start_time2)
                #input('')
                print('Number of MMPDE5 steps:', numMMPDEsteps)

                # Remesh via Galerkin projection
                #un_coeffs_MA = remesh_1d(un_coeffs_MA.clone().detach(), MA_meshpoints_old, num_meshpoints, MA_meshpoints, num_meshpoints, 10000,False)

                # Remeshing via interpolation
                spline_remesh = UnivariateSpline(MA_meshpoints_old.detach().numpy(), un_coeffs_MA.detach().numpy(),
                                                 s=0)  # s=0 ensures it interpolates through the data points
                un_coeffs_MA = torch.tensor(spline_remesh(MA_meshpoints.detach().numpy()), dtype=un_coeffs_MA.dtype)


                if opt['plots_mesh_movement']:
                    MA_mesh_save[k, :] = MA_meshpoints.clone().detach()
                    timesteps[k] = opt['tau'] * k
                    k = k + 1

        #mesh_time_MA=opt['num_eval_time_steps']*opt['num_time_steps']/(opt['num_eval_time_steps']*opt['num_time_steps']-1)*mesh_time_MA

        L2_MA = F.mse_loss(sol_MA, sol_fine[l, j, :])

        if opt['plots_mesh_movement']:
            plt.figure()
            for j in range(num_meshpoints):
                plt.plot(timesteps[0:-1],MA_mesh_save[0:-1,j].detach().numpy())
            plt.xlabel('Time')
            plt.ylabel('Mesh points')
            plt.title('Evolution of MA mesh points')
            plt.show()

        MA_time = time.process_time()-start_time

        #3) Get the model deformed mesh from trained model

        # Start the timer
        start_time = time.process_time()
        mesh_time_ML = 0.0
        start_time1 = time.process_time()

        MLmodel_coords = model(data)
        MLmodel_coords = MLmodel_coords.squeeze()
        u0_coeffs_ML,_=get_Burgers_initial_coeffs(fine_mesh_points, num_fine_mesh_points, MLmodel_coords, num_meshpoints, u0, load_quad_points, opt)
        un_coeffs_ML = u0_coeffs_ML.clone().detach()

        mesh_time_ML += 0.0#time.process_time()- start_time1

        if opt['plots_mesh_movement']:
            ML_mesh_save=torch.zeros(opt['num_eval_time_steps']*opt['num_time_steps'],num_meshpoints)
            timesteps=torch.zeros(opt['num_eval_time_steps']*opt['num_time_steps'])
            k=0
        #try:
        with torch.no_grad():
            for l in range(opt['num_eval_time_steps']-1):
                print(f"Time step {l}")
                MLmodel_coords_old = MLmodel_coords.clone().detach()
                for j in range(opt['num_time_steps']):
                    un_coeffs_ML, _, sol_ML, _, _=torch_FEM_Burgers_1D(opt, MLmodel_coords, quad_points, num_meshpoints, un_coeffs_ML.clone().detach())

                data.uu_tensor = un_coeffs_ML.clone().detach()
                data.x_phys=MLmodel_coords
                #data.pde_params['centers'][0][0][0]=np.random.randn(1)[0]

                #data.uu_tensor = torch.zeros_like(un_coeffs_ML)

                start_time1 = time.process_time()

                MLmodel_coords_old=MLmodel_coords.clone().detach()
                MLmodel_coords = model(data)
                MLmodel_coords = MLmodel_coords.clone().squeeze()
                #print(torch.norm(MLmodel_coords-MLmodel_coords_old))
                mesh_time_ML += time.process_time() - start_time1
                #print('Mesh time:', mesh_time_ML)
                #print('increment in time:', time.process_time() - start_time1)
                #input('')
                if opt['plots_mesh_movement']:
                    ML_mesh_save[k,:]=MLmodel_coords.clone().detach()
                    timesteps[k]=opt['tau']*k
                    k=k+1

                #un_coeffs_ML=remesh_1d(un_coeffs_ML.clone().detach(), MLmodel_coords_old, num_meshpoints, MLmodel_coords, num_meshpoints,load_quad_points)
                # with torch.no_grad():
                #     un_coeffs_ML = fn_expansion(un_coeffs_ML.clone().detach(), MLmodel_coords_old, MLmodel_coords, num_meshpoints)
                # Remeshing via interpolation
                spline_remesh = UnivariateSpline(MLmodel_coords_old.detach().numpy(), un_coeffs_ML.detach().numpy(),
                                                 s=0)  # s=0 ensures it interpolates through the data points
                un_coeffs_ML = torch.tensor(spline_remesh(MLmodel_coords.detach().numpy()), dtype=un_coeffs_ML.dtype)

        #torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints, un_coeffs, BC1=None, BC2=None)

        # End the timer
        MLmodel_time = time.process_time()-start_time

        #MLmodel_coords = MLmodel_coords.squeeze()

        #evaluate error
        L2_MLmodel = F.mse_loss(sol_ML,sol_fine[l,j,:])

        if opt['plots_mesh_movement']:
            plt.figure()
            for j in range(num_meshpoints):
                plt.plot(timesteps[0:-1],ML_mesh_save[0:-1,j].detach().numpy())
            plt.xlabel('Time')
            plt.ylabel('Mesh points')
            plt.title('Evolution of ML model mesh points')
            plt.show()

        # except:
        #     L2_MLmodel=L2_grid
        #     MLmodel_time=0.0
        #     mesh_time_ML=0.0
        #     print('Error in ML model evaluation')
        #     input('Press Enter to continue...')

        # Calculate error reduction ratios
        L2_reduction_MA = calculate_error_reduction(L2_grid, L2_MA)
        L2_reduction_MLmodel = calculate_error_reduction(L2_grid, L2_MLmodel)

        results.append({
            #'L1_grid': 0.0,
            'L2_grid': L2_grid.item(),
            # 'L1_MA': L1_MA,
            'L2_MA': L2_MA.item(),
            #'L1_MLmodel': 0.0,
            'L2_MLmodel': L2_MLmodel.item(),
            # 'L1_reduction_MA': L1_reduction_MA,
            'L2_reduction_MA': L2_reduction_MA.item(),
            #'L1_reduction_MLmodel': 0.0,
            'L2_reduction_MLmodel': L2_reduction_MLmodel.item()
        })

        times.append({
            'MA_time': MA_time,
            'MA_mesh_time': mesh_time_MA,
            'MLmodel_time': MLmodel_time,
            'ML_mesh_time': mesh_time_ML})

    df = pd.DataFrame(results)
    df_time = pd.DataFrame(times)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.describe())
    print(df_time.describe())

    return df, df_time


def plot_trained_dataset_1d_burgers(dataset, model, opt, model_out=None, show_mesh_evol_plots=False):
    mesh = dataset.mesh
    dim = len(dataset.opt['mesh_dims'])
    num_meshpoints = dataset.opt['mesh_dims'][0] if dim == 1 else dataset.opt['mesh_dims'][0] * dataset.opt['mesh_dims'][1]
    fine_mesh = UnitIntervalMesh(opt['num_fine_mesh_points'] - 1, name="fine_mesh")
    quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

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


        # Regular grid solution
        grid_coords = torch.linspace(0, 1, num_meshpoints)
        u0,u1_fine,u1_mesh,points=plotting_data_burgers(opt, grid_coords, quad_points, c_list, s_list)

        # 1) plot the FEM on regular mesh
        legend_list=[]
        axs0[i].plot(quad_points.detach().numpy(), u0,color='orange', linestyle='--')
        legend_list.append('Initial value')
        axs0[i].plot(quad_points.detach().numpy(), u1_fine, color='green')
        legend_list.append('True evolved value')
        axs0[i].plot(quad_points.detach().numpy(), u1_mesh, color='lightblue', linestyle='--')
        legend_list.append('Approximation')
        axs0[i].scatter(points, 0.0 * points, label='Mesh points', color='blue', marker='o')
        legend_list.append('Mesh points')
        axs0[i].legend(legend_list)

        # 2) MMPDE5 Mesh
        deformed_mesh = dataset.mesh_deformed
        deformed_mesh_points = torch.tensor(deformed_mesh.coordinates.dat.data_ro)

        u0, u1_fine, u1_mesh, points = plotting_data_burgers(opt, deformed_mesh_points, quad_points, c_list, s_list)

        legend_list=[]
        axs1[i].plot(quad_points.detach().numpy(), u0,color='orange', linestyle='--')
        legend_list.append('Initial value')
        axs1[i].plot(quad_points.detach().numpy(), u1_fine, color='green')
        legend_list.append('True evolved value')
        axs1[i].plot(quad_points.detach().numpy(), u1_mesh, color='lightblue', linestyle='--')
        legend_list.append('Approximation')
        axs1[i].scatter(points, 0.0 * points, label='Mesh points', color='blue', marker='o')
        legend_list.append('Mesh points')
        axs1[i].legend(legend_list)

        # 3) Get the model deformed mesh from trained model
        MLmodel_coords = model(data).detach()
        MLmodel_coords = MLmodel_coords.squeeze()

        u0, u1_fine, u1_mesh, points = plotting_data_burgers(opt, MLmodel_coords, quad_points, c_list, s_list)

        legend_list = []
        axs2[i].plot(quad_points.detach().numpy(), u0, color='orange', linestyle='--')
        legend_list.append('Initial value')
        axs2[i].plot(quad_points.detach().numpy(), u1_fine, color='green')
        legend_list.append('True evolved value')
        axs2[i].plot(quad_points.detach().numpy(), u1_mesh, color='lightblue', linestyle='--')
        legend_list.append('Approximation')
        axs2[i].scatter(points, 0.0 * points, label='Mesh points', color='blue', marker='o')
        legend_list.append('Mesh points')
        axs2[i].legend(legend_list)

    if opt['show_plots']:
        plt.show()
        fig0.savefig('../output/regular_mesh_burgers.png')
        fig1.savefig('../output/MMPDE5_mesh_burgers.png')
        fig2.savefig('../output/MLmodel_mesh_burgers.png')
