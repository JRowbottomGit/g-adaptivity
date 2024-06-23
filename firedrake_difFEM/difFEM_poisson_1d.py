import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.utils_main import plot_training_evol, plot_mesh_evol


# %% Define FEM basis functions (linear & piecewise cts splines)
def aux(x, a, b):  # Auxiliary function defining value in a single element
    return ((x >= torch.min(a, b)) * 1.0) * ((x <= torch.max(a, b)) * 1.0) * (x - a) / (b - a)

def daux(x, a, b):  # Auxiliary function defining derivative value in a single element
    return ((x >= torch.min(a, b)) * 1.0) * ((x <= torch.max(a, b)) * 1.0) * 1.0 / (b - a)

def phim(x, mesh, n):  # Basis functions
    if n == 0:
        return aux(x, mesh[n + 1], mesh[n])
    elif n == mesh.size()[0] - 1:
        return aux(x, mesh[n - 1], mesh[n])
    else:
        return aux(x, mesh[n - 1], mesh[n]) + aux(x, mesh[n + 1], mesh[n]) - ((x == mesh[n]) * 1.0)

def dphim(x, mesh, n):  # Derivative of basis functions
    if n == 0:
        return daux(x, mesh[n + 1], mesh[n])
    elif n == mesh.size()[0] - 1:
        return daux(x, mesh[n - 1], mesh[n])
    else:
        return daux(x, mesh[n - 1], mesh[n]) + daux(x, mesh[n + 1], mesh[n])

def f(x, c_list, s_list): # Forcing in Laplace's equation
    sol = torch.zeros_like(x)
    for c, s in zip(c_list, s_list):
        sol += -2 * torch.exp(-(x - c)**2 / s**2) * (s**2 - 2 * (x - c)**2) / s**4
    return sol

def u_true_exact_1d(x, c_list, s_list): # True solution
    if isinstance(x, torch.Tensor):
        sol = torch.zeros(x.shape[0])
        for c, s in zip(c_list, s_list):
            sol += torch.exp(-(x - c)**2 / s**2)
        return sol
    elif isinstance(x, np.ndarray):
        sol = np.zeros(x.shape[0])
        for c, s in zip(c_list, s_list):
            sol += np.exp(-(x - c)**2 / s**2)
        return sol

def u_true_exact_1d_vec(x, c_list, s_list):  # True solution
    if isinstance(x, torch.Tensor):
        sol = torch.zeros(x.shape)
        for c, s in zip(c_list, s_list):
            sol += torch.exp(-(x - c) ** 2 / s ** 2)
        return sol
    elif isinstance(x, np.ndarray):
        sol = np.zeros(x.shape)
        for c, s in zip(c_list, s_list):
            sol += np.exp(-(x - c) ** 2 / s ** 2)
        return sol

def soln(out, mesh, BC1, BC2, quad_points, num_solpoints): # Compute solution values of our approximation
    #function to do cummalative linear interpolation
    out = out.squeeze()
    # Extend the 'out' array to include boundary conditions
    extended_out = torch.cat([BC1, out, BC2])
    # Calculate gradients (dy/dx) for each interval in the mesh
    dy = extended_out[1:] - extended_out[:-1]
    dx = mesh[1:] - mesh[:-1]
    gradients = dy / dx
    # Find indices corresponding to the intervals for each point in 'x'
    indices = torch.searchsorted(mesh, quad_points, right=False) - 1
    indices = torch.clamp(indices, 0, num_solpoints - 1)
    # Get the starting point 'a' and gradient for each interval
    a = mesh[indices]
    grad = gradients[indices]
    # Perform linear interpolation
    sol = extended_out[indices] + grad * (quad_points - a)
    return sol

#%% L2 error
def L2norm(u, x): # Compute L2 norm of the function values u.
    return torch.trapezoid(abs(u)**2,x)

def build_stiffness_matrix(mesh_points, quad_points, num_meshpoints, stiff_quad_points=3, vectorise=True):
    if vectorise:
        # FOR OFF-DIAGONAL ELEMENTS (N-2 of them)
        # define quadrature points in overlapping support of basis functions
        k = stiff_quad_points
        mesh_diffs = torch.diff(mesh_points)
        if torch.any(mesh_diffs < 0.):
            print("WARNING: negative diffs in build_stiffness_matrix")
        L_start = mesh_points[:-1].view(-1, 1)
        mesh_diffs_quad = torch.arange(k + 1).view(1, -1).repeat(L_start.shape[0], 1) * mesh_diffs.view(-1, 1) / k
        mesh_quad = L_start + mesh_diffs_quad

        #compute product of derivatives of basis functions at quadrature points
        a = mesh_points[:-1]
        b = mesh_points[1:]
        #L_phi is the derivative of the left side of basis function in an internal, ie the positive slope
        L_dphi = (1 / (b - a)).view(-1, 1).repeat(1, k + 1)
        #R_phi is the derivative of the right side of basis function in an internal, ie the negative slope
        R_dphi = -L_dphi
        # compute stiffness matrix entries (N-1 of them)
        off_diags = torch.trapezoid(L_dphi * R_dphi, mesh_quad)

        # FOR INTERNAL DIAGONAL ELEMENTS (N-2 of them)
        diag_LHS = torch.trapezoid(L_dphi[:-1] ** 2, mesh_quad[:-1])
        diag_RHS = torch.trapezoid(R_dphi[1:] ** 2, mesh_quad[1:])
        internal_diag = diag_LHS + diag_RHS

        # FOR LEFT AND RIGHT BOUNDARY ELEMENTS
        mesh_quad_LHS = mesh_quad[0]
        mesh_quad_RHS = mesh_quad[-1]
        values_LHS = L_dphi[0]
        values_RHS = R_dphi[-1]
        LHS = torch.trapezoid(values_LHS ** 2, mesh_quad_LHS)
        RHS = torch.trapezoid(values_RHS ** 2, mesh_quad_RHS)

        # 4) combine
        A = torch.zeros(mesh_points.shape[0], mesh_points.shape[0])
        A[1:-1,1:-1] =  torch.diag(internal_diag)
        A += torch.diag(off_diags, 1) #upper diagonal
        A += torch.diag(off_diags, -1) #lower diagonal
        A[0, 0] = LHS
        A[-1, -1] = RHS
        return A

    else:
        A = torch.zeros(num_meshpoints, num_meshpoints)  # Stiffness matrx
        for m in range(num_meshpoints):
            for n in range(num_meshpoints):
                A[m, n] = -torch.trapezoid(dphim(quad_points, mesh_points, m + 1) * dphim(quad_points, mesh_points, n + 1), quad_points)

    return A


def build_load_vector(mesh, x, BC1, BC2, num_meshpoints, c_list, s_list, load_quad_points):

    RHS = torch.zeros(num_meshpoints)  # Forcing
    k = load_quad_points #number of quadrature points per interval
    diffs = torch.diff(mesh)
    L_start = mesh[:-1].view(-1, 1)
    phis = torch.arange(k).view(1, -1) / (k-1)
    x_diffs = diffs.view(-1, 1) * torch.arange(k).view(1, -1).repeat(L_start.shape[0], 1) / (k-1)
    x_vec = L_start + x_diffs
    f_vec = f(x_vec, c_list, s_list)

    #integral of product with basis functions
    left_interval_ints = torch.trapezoid(f_vec * phis, x_vec)
    reversed_phis = torch.flip(phis, dims=[1])
    right_interval_ints = torch.trapezoid(f_vec * reversed_phis, x_vec)

    #collect contributions from left and right including boundary nodes
    RHS[1:] += left_interval_ints#[0:-1] #contributions from left interval
    RHS[:-1] += right_interval_ints#[1:] #contributions from right interval

    return RHS.unsqueeze(-1)


def gradient_meshpoints_1D(opt,data, x_phys):
    if 'grad_type' not in opt.keys():
        error_message = "Error: opt['grad_type'] not specified"
        raise ValueError(error_message)
    else:
        if opt['grad_type']=='PDE_loss_direct_mse':
            return gradient_meshpoints_1D_PDE_loss_direct_mse(opt,data, x_phys)
        elif opt['grad_type']=='PDE_loss_direct_L2':
            return gradient_meshpoints_1D_PDE_loss_direct_L2(opt,data, x_phys)
        elif opt['grad_type']=='burgers_timestep_loss_direct_mse':
            return gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt,data, x_phys)#(opt, mesh_points, quad_points, num_meshpoints, un_coeffs)
        else:
            error_message = "Error: opt['grad_type'] incorrectly specified"
        raise ValueError(error_message)

def gradient_meshpoints_1D_PDE_loss_direct_L2(opt,data, x_phys):
    mesh_points = torch.tensor(x_phys.detach().numpy(), requires_grad=True)
    # Extract some relevant parameters
    num_meshpoints = opt['mesh_dims'][0]
    c_list=[torch.tensor(c) for c in data.pde_params['centers'][0]]
    s_list=[torch.tensor(s) for s in data.pde_params['scales'][0]]

    # %% Assembly of linear system
    quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

    out, mesh_points_fem, sol, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list)

    # %% Compute Loss (L2 error)
    loss = L2norm(sol - u_true_exact_1d(quad_points, c_list, s_list), quad_points)
    loss.backward()
    return torch.tensor(loss.detach().numpy()), mesh_points.grad


def gradient_meshpoints_1D_PDE_loss_direct_mse(opt,data, x_phys):
    mesh_points = torch.tensor(x_phys.detach().numpy(), requires_grad=True)
    # Extract some relevant parameters
    num_meshpoints = opt['mesh_dims'][0]
    c_list=[torch.tensor(c) for c in data.pde_params['centers'][0]]
    s_list=[torch.tensor(s) for s in data.pde_params['scales'][0]]

    # %% Assembly of linear system
    quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

    out, mesh_points_fem, sol, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list)

    # %% Compute Loss (L2 error)
    loss = F.mse_loss(sol, u_true_exact_1d(quad_points, c_list, s_list))
    loss.backward()
    return torch.tensor(loss.detach().numpy()), mesh_points.grad

def torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list):
    load_quad_points = opt['load_quad_points']
    stiff_quad_points = opt['stiff_quad_points']

    internal_mesh_points = mesh_points[1:-1]
    num_internal_meshpoints = internal_mesh_points.shape[0]
    # build a (N)x(N) stiffness matrix, internal is (N-1)x(N-1) but need corners for load vector adjustment terms
    A = build_stiffness_matrix(mesh_points, quad_points, num_internal_meshpoints, stiff_quad_points, vectorise=True, keyops=False)
    A_int = -A[1:-1, 1:-1]  # remove the first and last rows and columns
    # build a (N-1)x1 internal load vector
    BC1 = u_true_exact_1d(torch.tensor([mesh_points[0]]), c_list, s_list)  # 0.0 # Left endpoint Dirichlet BC
    BC2 = u_true_exact_1d(torch.tensor([mesh_points[-1]]), c_list, s_list)  # 0.0 # Right endpoint Dirichlet BC
    RHS = build_load_vector(mesh_points, quad_points, BC1, BC2, num_meshpoints, c_list, s_list, load_quad_points)
    RHS_int = RHS[1:-1]  # remove the first and last rows

    # adjust the load vector to account for the BCs
    # specifically this means adjust the 1st and last (9th) entries of the load vector
    # to account for interactions with the basis functions of the BCs at nodes 0 and 10
    adj_bc1 = BC1 * A[0, 1]
    RHS_int[0] += adj_bc1
    adj_bc2 = A[-1, -2] * BC2
    RHS_int[-1] += adj_bc2

    # %% Solution of linear system
    coeffs = torch.linalg.solve(A_int, RHS_int)
    sol = soln(coeffs, mesh_points, BC1, BC2, quad_points, num_solpoints=num_meshpoints)

    return coeffs, mesh_points, sol, BC1, BC2


def train_step_vec(opt, num_meshpoints, lr, epochs, c_list, s_list, plotandsave=False):

    if opt['mesh_params'] == "all":
        mesh_points = torch.tensor(np.linspace(0, 1, num_meshpoints), requires_grad=True)
        optimizer = torch.optim.SGD([mesh_points], lr=lr)

    elif opt['mesh_params'] == "internal":
        # %% Initial mesh
        mesh_points_internal= torch.linspace(1.0/(num_meshpoints-1), 1-1.0/(num_meshpoints-1), num_meshpoints-2, requires_grad=True)
        mesh_points = torch.cat((torch.tensor([0.0]), mesh_points_internal, torch.tensor([1.0])), dim=0)
        optimizer = torch.optim.SGD([mesh_points_internal], lr=lr)

    loss_list, mesh_list = [], []
    for j in range(epochs):
        optimizer.zero_grad()

        # %% Assembly of linear system
        quad_points = torch.linspace(0, 1, opt['eval_quad_points'])
        out, mesh_points_fem, sol, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list)

        # %% Compute Loss (L2 error)
        loss = L2norm(sol - u_true_exact_1d(quad_points, c_list, s_list), quad_points)
        print("Iteration:", j, "Loss:", loss.item())
        loss.backward()

        # Update internal mesh points
        optimizer.step()

        if opt['mesh_params'] == "internal":
            # Update total mesh points
            mesh_points = torch.cat((torch.tensor([0.0]), mesh_points_internal, torch.tensor([1.0])), dim=0)
        elif opt['mesh_params'] == "all":
            #%% Post-process: Rescale, Resort, and Clip endpoints
            with torch.no_grad():
                new_val = mesh_points
                new_val = (new_val - min(new_val)) / (max(new_val) - min(new_val))  # Rescale
                new_val[0] = 0.0  # Clip endpoints
                new_val[-1] = 1.0
                mesh_points.copy_(new_val)

        loss_list.append(loss.item())
        mesh_list.append(mesh_points.detach().numpy().copy()) #requires .copy as .copy_ and .add_ are in-place operations

        #%% Plot the results
        if plotandsave and j%5==1:
            plot_and_save(quad_points.detach().numpy(),
                          u_true=u_true_exact_1d(quad_points, c_list, s_list).detach().numpy(),
                          u_approx=soln(out, mesh_points, BC1, BC2, quad_points, num_meshpoints).detach().numpy(),
                          mesh=mesh_points.detach().numpy(),
                          save_str=f'./movie/frame_{j:03d}.png', show=True)

    return out, mesh_points, sol, loss_list, mesh_list


class backFEM_1D(torch.nn.Module):
    '''a wrapper for differentiable backFEM solver that minimizes the L2 error of the approximation and returns the updated mesh'''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_meshpoints = opt['mesh_dims'][0]
        self.eval_quad_points = opt['eval_quad_points']
        self.lr = opt['lr']
        self.epochs = opt['epochs']
        # self.c = opt['center']
        # self.s = opt['scale']
        self.plot_evol_flag = False
        if self.opt['show_train_evol_plots']:
            self.loss_fig, loss_axs = plt.subplots(3, 3, figsize=(15, 15))
            self.loss_axs = loss_axs.ravel()
        if self.opt['show_mesh_evol_plots']:
            self.mesh_fig, mesh_axs = plt.subplots(3, 3, figsize=(15, 15))
            self.mesh_axs = mesh_axs.ravel()

    def forward(self, data):
        c_list = [torch.from_numpy(c) for c in data.pde_params['centers'][0]]
        s_list = [torch.from_numpy(s) for s in data.pde_params['scales'][0]]
        coeffs, MLmodel_coords, sol, loss_list, mesh_list = train_step_vec(self.opt, self.num_meshpoints, self.lr, self.epochs, c_list, s_list)

        return coeffs, MLmodel_coords, sol


class Fixed_Mesh_1D(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_meshpoints = opt['mesh_dims'][0]

    def forward(self, data):
        mesh_points = data.x_comp
        if self.opt['loss_type'] == 'mesh_loss':
            return mesh_points

        elif self.opt['loss_type'] == 'pde_loss':
            pass

def plot_and_save(x, u_true=None, u_approx=None, mesh=None, save_str=None, show=True):
    legend_list = []
    if u_true is not None:
        plt.plot(x, u_true)
        legend_list.append('True value')
    if u_approx is not None:
        plt.plot(x, u_approx)
        legend_list.append('Approximation')
    if mesh is not None:
        plt.scatter(mesh, 0.0 * mesh, label='Mesh points', color='blue', marker='o')
        legend_list.append('Mesh points')
    plt.legend(legend_list)

    if save_str is not None:
        plt.savefig(save_str)

    if show:
        plt.show()

def plot_and_save_Burgers(x, u_init=None,u_true=None, u_approx=None, mesh=None, save_str=None, show=True):
    legend_list = []
    if u_true is not None:
        plt.plot(x, u_init, linestyle='dashed')
        legend_list.append('Initial value')
    if u_true is not None:
        plt.plot(x, u_true)
        legend_list.append('True evolved value')
    if u_approx is not None:
        plt.plot(x, u_approx)
        legend_list.append('Approximation')
    if mesh is not None:
        plt.scatter(mesh, 0.0 * mesh, label='Mesh points', color='blue', marker='o')
        legend_list.append('Mesh points')
    plt.legend(legend_list)

    if save_str is not None:
        plt.savefig(save_str)

    if show:
        plt.show()


##  DIFFERENTIABLE BURGERS TIMESTEPPER

def fn_expansion(coeffs, mesh,quad_points, num_solpoints): # Compute solution values of our approximation
    # Calculate gradients (dy/dx) for each interval in the mesh
    dy = coeffs[1:] - coeffs[:-1]
    dx = mesh[1:] - mesh[:-1]
    gradients = dy / dx
    gradients = torch.cat((gradients, torch.tensor([0.0])),dim=0)
    # Find indices corresponding to the intervals for each point in 'x'
    indices = torch.searchsorted(mesh, quad_points, right=False) - 1
    indices = torch.clamp(indices, 0, num_solpoints - 1)
    # Get the starting point 'a' and gradient for each interval
    a = mesh[indices]
    grad = gradients[indices]
    # Perform linear interpolation
    sol = coeffs[indices] + grad * (quad_points - a)
    return sol

def dxfn_expansion(coeffs, mesh,quad_points, num_solpoints): # Compute solution values of our approximation
    # Calculate gradients (dy/dx) for each interval in the mesh
    dx = mesh[1:] - mesh[:-1]

    dphi = 1/dx
    coeffs_l=coeffs[:-1]
    coeffs_r=coeffs[1:]

    a=coeffs_r*dphi-coeffs_l*dphi

    # Find indices corresponding to the intervals for each point in 'x'
    indices = torch.searchsorted(mesh, quad_points, right=False) - 1
    indices = torch.clamp(indices, 0, num_solpoints - 1)

    # Perform linear interpolation
    sol = a[indices]
    return sol

def fast_inner_product(mesh, f ,load_quad_points,num_meshpoints):

    RHS = torch.zeros(num_meshpoints)  # Forcing

    k = load_quad_points #number of quadrature points per interval
    diffs = torch.diff(mesh)
    L_start = mesh[:-1].view(-1, 1)
    phis = torch.arange(k).view(1, -1) / (k-1)
    x_diffs = diffs.view(-1, 1) * torch.arange(k).view(1, -1).repeat(L_start.shape[0], 1) / (k-1)
    x_vec = L_start + x_diffs
    f_vec = f(x_vec)

    #integral of product with basis functions
    left_interval_ints = torch.trapezoid(f_vec * phis, x_vec)
    reversed_phis = torch.flip(phis, dims=[1])
    right_interval_ints = torch.trapezoid(f_vec * reversed_phis, x_vec)

    #collect contributions from left and right including boundary nodes
    RHS[1:] += left_interval_ints#[0:-1] #contributions from left interval
    RHS[:-1] += right_interval_ints#[1:] #contributions from right interval

    return RHS.unsqueeze(-1)

def gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt,data, x_phys):
    ## TODO!
    mesh_points = torch.tensor(x_phys.detach().numpy(), requires_grad=True)
    # Extract some relevant parameters
    num_meshpoints = opt['mesh_dims'][0]
    num_quad_points = opt['load_quad_points']
    load_quad_points = opt['load_quad_points']

    # Initialise fine mesh points
    num_fine_meshpoints = opt['num_fine_mesh_points']
    fine_mesh_points = torch.linspace(0, 1, opt['num_fine_mesh_points'],dtype=mesh_points.dtype)

    # %% Assembly of linear system
    quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

    # Initialize the data dictionary
    c_list=[torch.tensor(c) for c in data.pde_params['centers'][0]]
    s_list=[torch.tensor(s) for s in data.pde_params['scales'][0]]

    u0 = lambda x: opt['gauss_amplitude']*u_true_exact_1d_vec(x, c_list, s_list)
    u0_coeffs, u0_coeffs_fine = get_Burgers_initial_coeffs(fine_mesh_points, num_fine_meshpoints, mesh_points,num_meshpoints, u0, load_quad_points, opt)

    # Time-stepping on fine grid
    with torch.no_grad():
        un_coeffs_fine = u0_coeffs_fine.clone().detach()
        for j in range(opt['num_time_steps']):
            un_coeffs_fine, _, sol_fine, BC1, BC2 = torch_FEM_Burgers_1D(opt, fine_mesh_points, quad_points, num_fine_meshpoints, un_coeffs_fine)

    # Time-stepping of on coarse grid
    un_coeffs=u0_coeffs
    for j in range(opt['num_time_steps']):
        un_coeffs, _, sol, BC1, BC2 = torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints, un_coeffs)

    # %% Compute Loss (L2 error)
    loss = F.mse_loss(sol, sol_fine)
    loss.backward()
    return torch.tensor(loss.detach().numpy()), mesh_points.grad

# For testing purposes we initialise a data object:
class Data:
    def __init__(self, pde_params):
        self.pde_params = pde_params

def build_mass_matrix(mesh_points, load_quad_points,num_meshpoints):
    M = torch.zeros(num_meshpoints, num_meshpoints)  # Mass matrix

    for n in range(num_meshpoints):
        def phi(x):
            return phim(x, mesh_points, n)

        M[:, n] = fast_inner_product(mesh_points, phi, load_quad_points,num_meshpoints)[:, 0]
    return M

def remesh_1d(un_coeffs,mesh_points, num_meshpoints, fine_mesh_points,num_fine_meshpoints, load_quad_points,enforce_BC=True):
    def un(x):
        return fn_expansion(un_coeffs, mesh_points, x, num_meshpoints)

    # Build mass matrix
    M = build_mass_matrix(mesh_points, load_quad_points, num_meshpoints)
    RHS = fast_inner_product(fine_mesh_points, un, load_quad_points, num_fine_meshpoints)[:, 0]

    if enforce_BC:
        # Impose BCs
        RHS[0] = un(torch.tensor([0.0]))
        RHS[-1] = un(torch.tensor([1.0]))
        # Correct Mass Matrx
        M[0, :] = torch.zeros(num_meshpoints)
        M[-1, :] = torch.zeros(num_meshpoints)
        M[0, 0] = 1.0
        M[-1, -1] = 1.0

    # Find new expansion coefficients
    return torch.tensor(np.linalg.solve(M.detach().numpy(), RHS))#torch.cat((torch.tensor([un_coeffs[0]]),fast_inner_product(fine_mesh_points, un, load_quad_points,num_fine_meshpoints)[1:-1, 0],torch.tensor([un_coeffs[-1]])),dim=0)

def torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints, un_coeffs, BC1=None, BC2=None):
    # Check if BC1, BC2 provided
    if BC1 is None:
        BC1 = un_coeffs[0]
    if BC2 is None:
        BC2 = un_coeffs[-1]

    # Parameters
    load_quad_points = opt['load_quad_points']
    stiff_quad_points = opt['stiff_quad_points']
    num_sol_points = opt['eval_quad_points']
    tau=opt['tau']
    nu=opt['nu']

    # Assembly mass matrix M and stiffness matrix A

    A = torch.zeros(num_meshpoints, num_meshpoints)  # Stiffness matrx
    M = build_mass_matrix(mesh_points, load_quad_points,num_meshpoints)
    A = build_stiffness_matrix(mesh_points, quad_points, load_quad_points)

    def un_dun(x):
        return (fn_expansion(un_coeffs, mesh_points, x, num_meshpoints)) * dxfn_expansion(un_coeffs,
                                                                                          mesh_points, x,
                                                                                          num_meshpoints)

    RHS = torch.matmul(M, un_coeffs) - tau * fast_inner_product(mesh_points, un_dun, load_quad_points,num_meshpoints)[:, 0]

    # Correct for BCs

    RHS[0] = BC1
    RHS[-1] = BC2

    Matrix = M + tau * nu * A

    # Correct matrix for BCs
    Matrix[0, :] = torch.zeros(num_meshpoints)
    Matrix[-1, :] = torch.zeros(num_meshpoints)
    Matrix[0, 0] = 1.0
    Matrix[-1, -1] = 1.0

    unp1_coeffs = torch.linalg.solve(Matrix, RHS)
    sol = fn_expansion(unp1_coeffs, mesh_points, quad_points, num_meshpoints)

    return unp1_coeffs, mesh_points, sol, BC1, BC2

def train_step_Burgers(opt, num_meshpoints, lr, epochs, c_list, s_list, plotandsave=False):

    # %% Initial mesh
    ##mesh_points_old = torch.tensor(np.linspace(0, 1, num_meshpoints), requires_grad=True)
    #mesh_points_internal= torch.linspace(1.0/(num_meshpoints-1), 1-1.0/(num_meshpoints-1), num_meshpoints-2, requires_grad=True)
    #mesh_points = torch.cat((torch.tensor([0.0]), mesh_points_internal, torch.tensor([1.0])), dim=0)

    mesh_points_internal = torch.linspace(1.0 / (num_meshpoints - 1), 1 - 1.0 / (num_meshpoints - 1),
                                          num_meshpoints - 2,
                                          requires_grad=True)
    mesh_points = torch.cat((torch.tensor([0.0]), mesh_points_internal, torch.tensor([1.0])), dim=0)

    # optimizer = torch.optim.Adam([mesh], lr=lr)
    optimizer = torch.optim.SGD([mesh_points_internal], lr=lr)

    loss_list, mesh_list = [], []
    for j in range(epochs):

        optimizer.zero_grad()

        # %% Assembly of linear system
        quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

        #out, mesh_points_fem, sol, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list)
        x_phys=mesh_points.clone().detach()

        example_pde_params = {'centers': [c_list], 'scales': [s_list]}
        # Initialize the data object
        data = Data(pde_params=example_pde_params)
        # Compute the loss and the gradient
        loss, grads = gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt, data, x_phys)
        # %% Compute Loss (L2 error)
        aux_loss=torch.sum(mesh_points*grads)
        print("Iteration:", j, "Loss:", loss.item(), "Norm Grad:", torch.norm(grads).item())

        aux_loss.backward()

        # Update internal mesh points
        optimizer.step()

        # Update total mesh points
        mesh_points = torch.cat((torch.tensor([0.0]), mesh_points_internal, torch.tensor([1.0])), dim=0)

        # # Update internal mesh points
        # with torch.no_grad():
        #     mesh_points -= lr * grads
        #
        #
        # # Update total mesh points
        # #mesh_points = torch.cat((torch.tensor([0.0]), mesh_points[1:-1], torch.tensor([1.0])), dim=0)
        # mesh_points[0]=0.0
        # mesh_points[-1]=1.0

        loss_list.append(loss.item())
        mesh_list.append(mesh_points.detach().numpy().copy()) #requires .copy as .copy_ and .add_ are in-place operations

        #%% Plot the results
        if plotandsave and j%(epochs-2)==1:
            u0 = lambda x: opt['gauss_amplitude']*u_true_exact_1d_vec(x, c_list, s_list)
            # u0 = data['u0']

            num_fine_meshpoints=opt['num_fine_meshpoints']
            fine_mesh_points = torch.linspace(0, 1, num_fine_meshpoints)

            # M_fine = build_mass_matrix(fine_mesh_points, 10*opt['eval_quad_points'], num_fine_meshpoints)
            # RHS_fine=fast_inner_product(fine_mesh_points, u0, load_quad_points,num_fine_meshpoints)[:, 0].detach().numpy()
            # # Impose BCs
            # RHS_fine[0] = u0(torch.tensor([0.0]))
            # RHS_fine[-1] = u0(torch.tensor([1.0]))
            # # Correct Mass Matrx
            # M_fine[0, :] = torch.zeros(num_fine_meshpoints)
            # M_fine[-1, :] = torch.zeros(num_fine_meshpoints)
            # M_fine[0, 0] = 1.0
            # M_fine[-1, -1] = 1.0
            #
            # # Remesh to fine grid
            # u0_coeffs_fine = torch.tensor(np.linalg.solve(M_fine.detach().numpy(),RHS_fine))  # remesh_1d(u0_coeffs, mesh_points, num_meshpoints, fine_mesh_points, num_fine_meshpoints, num_quad_points)
            #
            # # Build mass matrix
            # M = build_mass_matrix(mesh_points, opt['eval_quad_points'], num_meshpoints)
            # RHS = fast_inner_product(mesh_points, u0, load_quad_points,num_meshpoints)[:, 0].detach().numpy()
            # # Impose BCs
            # RHS[0] = u0(torch.tensor([0.0]))
            # RHS[-1] = u0(torch.tensor([1.0]))
            # # Correct Mass Matrx
            # M[0, :] = torch.zeros(num_meshpoints)
            # M[-1, :] = torch.zeros(num_meshpoints)
            # M[0, 0] = 1.0
            # M[-1, -1] = 1.0
            #
            # # Find expansion coefficients for the initial condition
            # u0_coeffs = torch.tensor(np.linalg.solve(M.detach().numpy(),RHS))

            u0_coeffs, u0_coeffs_fine=get_Burgers_initial_coeffs(fine_mesh_points, num_fine_meshpoints, mesh_points, num_meshpoints, u0, load_quad_points, opt)

            # Time-stepping on fine grid
            with torch.no_grad():
                un_coeffs = u0_coeffs.clone().detach()
                un_fine_coeffs = u0_coeffs_fine.clone().detach()
                for j in range(opt['num_time_steps']):
                    un_coeffs, _, sol, BC1, BC2 = torch_FEM_Burgers_1D(opt, mesh_points, quad_points,num_meshpoints, un_coeffs)
                    un_fine_coeffs, _, sol_fine, BC1, BC2 = torch_FEM_Burgers_1D(opt, fine_mesh_points, quad_points,num_fine_meshpoints, un_fine_coeffs)

                plot_and_save_Burgers(quad_points.detach().numpy(),
                          u_init=u_true_exact_1d(quad_points, c_list, s_list).detach().numpy(),
                          u_true=sol_fine.detach().numpy(),
                          u_approx=sol.detach().numpy(),
                          mesh=mesh_points.detach().numpy(),
                          save_str=f'./movie/frame_{j:03d}.png', show=True)
    out=mesh_points.clone().detach()
    return out, mesh_points, sol, loss_list, mesh_list

def get_Burgers_initial_coeffs_old(fine_mesh_points, num_fine_meshpoints, mesh_points, num_meshpoints, u0, load_quad_points, opt):
    M_fine = build_mass_matrix(fine_mesh_points, 10 * opt['eval_quad_points'], num_fine_meshpoints)
    RHS_fine = fast_inner_product(fine_mesh_points, u0, load_quad_points, num_fine_meshpoints)[:, 0].detach().numpy()
    # Impose BCs
    RHS_fine[0] = u0(torch.tensor([0.0]))
    RHS_fine[-1] = u0(torch.tensor([1.0]))

    return u0(mesh_points),u0(fine_mesh_points)

def get_Burgers_initial_coeffs(fine_mesh_points, num_fine_meshpoints, mesh_points, num_meshpoints, u0, load_quad_points, opt):
    M_fine = build_mass_matrix(fine_mesh_points, 10 * opt['eval_quad_points'], num_fine_meshpoints)
    RHS_fine = fast_inner_product(fine_mesh_points, u0, load_quad_points, num_fine_meshpoints)[:, 0].detach().numpy()
    # Impose BCs
    RHS_fine[0] = u0(torch.tensor([0.0]))
    RHS_fine[-1] = u0(torch.tensor([1.0]))
    # Correct Mass Matrx
    M_fine[0, :] = torch.zeros(num_fine_meshpoints)
    M_fine[-1, :] = torch.zeros(num_fine_meshpoints)
    M_fine[0, 0] = 1.0
    M_fine[-1, -1] = 1.0

    # Remesh to fine grid
    u0_coeffs_fine = torch.tensor(np.linalg.solve(M_fine.detach().numpy(),
                                                  RHS_fine))  # remesh_1d(u0_coeffs, mesh_points, num_meshpoints, fine_mesh_points, num_fine_meshpoints, num_quad_points)

    # Build mass matrix
    M = build_mass_matrix(mesh_points, opt['eval_quad_points'], num_meshpoints)
    RHS = fast_inner_product(mesh_points, u0, load_quad_points, num_meshpoints)[:, 0].detach().numpy()
    # Impose BCs
    RHS[0] = u0(torch.tensor([0.0]))
    RHS[-1] = u0(torch.tensor([1.0]))
    # Correct Mass Matrx
    M[0, :] = torch.zeros(num_meshpoints)
    M[-1, :] = torch.zeros(num_meshpoints)
    M[0, 0] = 1.0
    M[-1, -1] = 1.0

    # Find expansion coefficients for the initial condition
    u0_coeffs = torch.tensor(np.linalg.solve(M.detach().numpy(), RHS))

    return u0_coeffs,u0_coeffs_fine

def plotting_data_burgers(opt,mesh_points,quad_points,c_list, s_list):
    u0 = lambda x: opt['gauss_amplitude']*u_true_exact_1d_vec(x, c_list, s_list)
    # u0 = data['u0']
    load_quad_points = opt['load_quad_points']
    num_fine_meshpoints = opt['num_fine_mesh_points']
    num_meshpoints = mesh_points.shape[0]

    fine_mesh_points = torch.linspace(0, 1, num_fine_meshpoints)
    # M_fine = build_mass_matrix(fine_mesh_points, 10 * opt['eval_quad_points'], num_fine_meshpoints)
    #
    # # Remesh to fine grid
    # u0_coeffs_fine = torch.tensor(np.linalg.solve(M_fine.detach().numpy(),
    #                                               fast_inner_product(fine_mesh_points, u0, load_quad_points,
    #                                                                  num_fine_meshpoints)[:,
    #                                               0].detach().numpy()))  # remesh_1d(u0_coeffs, mesh_points, num_meshpoints, fine_mesh_points, num_fine_meshpoints, num_quad_points)
    #
    # # Build mass matrix
    # M = build_mass_matrix(mesh_points, opt['eval_quad_points'], num_meshpoints)
    #
    # # Find expansion coefficients for the initial condition
    # u0_coeffs = torch.tensor(np.linalg.solve(M.detach().numpy(),
    #                                          fast_inner_product(mesh_points, u0, load_quad_points,
    #                                                             num_meshpoints)[:, 0].detach().numpy()))
    u0_coeffs, u0_coeffs_fine = get_Burgers_initial_coeffs(fine_mesh_points, num_fine_meshpoints, mesh_points,num_meshpoints, u0, load_quad_points, opt)

    # Time-stepping on fine grid
    with torch.no_grad():
        un_coeffs = u0_coeffs.clone().detach()
        un_fine_coeffs = u0_coeffs_fine.clone().detach()
        for j in range(opt['num_time_steps']):
            un_coeffs, _, sol, BC1, BC2 = torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints, un_coeffs)
            un_fine_coeffs, _, sol_fine, BC1, BC2 = torch_FEM_Burgers_1D(opt, fine_mesh_points, quad_points,
                                                                         num_fine_meshpoints, un_fine_coeffs)

    return opt['gauss_amplitude']*u_true_exact_1d(quad_points, c_list, s_list).detach().numpy(),sol_fine.detach().numpy(),sol.detach().numpy(),mesh_points.detach().numpy()


if __name__ == '__main__':
    # %% Parameters
    num_meshpoints = 15  # Means we have points x_n with n=0,1,...,N+1
    load_quad_points = 2000  # Number of quadrature points for FEM assembly
    tau = 0.006 #0.01#0.006 #2  # Gradient descent speed
    c_list = torch.tensor([np.array([0.2])])  # True value of c
    s_list = torch.tensor([np.array([0.2])])  # Scaling parameter

    # train_step(num_meshpoints, num_quad_points, tau, s)
    # train_step_optim(num_meshpoints, num_quad_points, tau, s)
    opt = {'show_mesh_evol_plots': False}
    opt['mesh_params'] = "internal"
    opt['eval_quad_points'] = 100
    opt['load_quad_points'] = 100
    opt['stiff_quad_points'] = 3
    opt['num_time_steps'] = 10
    opt['mesh_dims'] = [num_meshpoints]
    opt['tau']=0.01
    opt['nu']=0.005
    #train_step_vec(opt, num_meshpoints, 0.1,200, c_list, s_list, plotandsave=True)
    train_step_Burgers(opt, num_meshpoints, 0.2,100, c_list, s_list, plotandsave=True)
    #(opt, num_meshpoints, lr, epochs, c_list, s_list, plotandsave=False)
    # pykeops.test_numpy_bindings()  # perform the compilation
    # pykeops.test_torch_bindings()  # perform the compilation

    # # %% Parameters
    # num_meshpoints = 50  # Means we have points x_n with n=0,1,...,N+1
    # load_quad_points = 100  # Number of quadrature points for FEM assembly
    # num_eval_quad_points = 100  # Number of quadrature points for evaluation
    # tau = 0.01  # 0.01#0.006 #2  # Gradient descent speed
    # c_list = [np.array([0.5])]  # True value of c
    # s_list = [np.array([0.1])]  # Scaling parameter
    # mesh_points = torch.tensor(np.linspace(0, 1, num_meshpoints), requires_grad=True)
    # quad_points = torch.tensor(np.linspace(0, 1, load_quad_points), requires_grad=False)
    # nu = 0.005  # Viscosity
    #
    # # Assembly mass matrix M and stiffness matrix A
    #
    # M= build_mass_matrix(mesh_points, load_quad_points, num_meshpoints)
    #
    # #A = build_stiffness_matrix(mesh_points, quad_points, load_quad_points)
    #
    # c = 3.0
    # alpha = 4.0
    #
    #
    # # Initial condition
    # def u0(x):
    #     return 0.5 + 0.1 * torch.sin(2 * torch.pi * x)
    #     # return c-alpha*np.tanh(alpha*x/2.0/nu)
    #     # return -torch.sign(x-0.25)
    #     # return np.exp(-((x-c_list[0])**2)/s_list[0]**2)
    #
    #
    # # Find expansion coefficients for the initial condition
    # u0_coeffs = torch.tensor(np.linalg.solve(M.detach().numpy(),
    #                                          fast_inner_product(mesh_points, u0, load_quad_points, num_meshpoints)[:,
    #                                          0].detach().numpy()))
    #
    # # Check with plot
    # u0_expanded = fn_expansion(u0_coeffs, mesh_points, quad_points, num_eval_quad_points)
    # dxu0_expanded = dxfn_expansion(u0_coeffs, mesh_points, quad_points, num_eval_quad_points)
    # # Plotting of the initial condition
    # x = torch.tensor(np.linspace(0, 1, num_eval_quad_points))
    # plt.plot(x.detach().numpy(), u0(x).detach().numpy())
    # plt.plot(x.detach().numpy(), u0_expanded.detach().numpy())
    # # plt.plot(x, dxu0_expanded.detach().numpy())
    # plt.title('Initial condition')
    # plt.show()
    #
    #
    # un_coeffs = u0_coeffs.clone().detach()
    # # Boundary conditions
    # BC1 = u0(torch.tensor([0.0]))  # 0.0 # Left endpoint Dirichlet BC
    # BC2 = u0(torch.tensor([1.0]))
    #
    # # Define opt as a dictionary
    # opt = {}
    # opt['load_quad_points'] = 100
    # opt['stiff_quad_points'] = 100
    # opt['eval_quad_points'] = 100
    # opt['tau'] = 0.01
    # opt['nu'] = 0.005
    #
    # un_coeffs = u0_coeffs.clone().detach()
    # # Boundary conditions
    # BC1 = u0(torch.tensor([0.0]))  # 0.0 # Left endpoint Dirichlet BC
    # BC2 = u0(torch.tensor([1.0]))
    #
    # with torch.no_grad():
    #     for j in range(1000):  # Careful with the minus sign because we integrate by parts!
    #         # Use routine for Burger's time step
    #         un_coeffs1, mesh_points, sol, BC1, BC2 = torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints, un_coeffs)
    #         un_coeffs=un_coeffs1.clone().detach()
    #         if j % 20 == 0:
    #             un_expanded = fn_expansion(un_coeffs, mesh_points, quad_points, num_eval_quad_points)
    #             plt.plot(x, un_expanded.detach().numpy())
    #             plt.title('Evolution full Burgers2')
    #     plt.show()