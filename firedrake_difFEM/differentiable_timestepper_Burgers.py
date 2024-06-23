import time
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
from difFEM_poisson_1d import phim, dphim,build_stiffness_matrix, soln
# from torchquad import Trapezoid
# import pykeops
# from pykeops.torch import LazyTensor

from src.utils_main import plot_training_evol, plot_mesh_evol
def fn_expansion(coeffs, mesh,quad_points, num_solpoints): # Compute solution values of our approximation
    # Calculate gradients (dy/dx) for each interval in the mesh
    dy = coeffs[1:] - coeffs[:-1]
    dx = mesh[1:] - mesh[:-1]
    gradients = dy / dx
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

    # Initialise fine mesh points
    num_fine_meshpoints = opt['eval_quad_points']
    fine_mesh_points = torch.linspace(0, 1, opt['eval_quad_points'],dtype=torch.float64)

    # %% Assembly of linear system
    quad_points = torch.linspace(0, 1, opt['eval_quad_points'])

    # Initial condition from data

    u0 = data['u0']

    # Find expansion coefficients for the initial condition
    u0_coeffs = torch.tensor(np.linalg.solve(M.detach().numpy(),fast_inner_product(mesh_points, u0, load_quad_points,num_meshpoints)[:,0].detach().numpy()))

    # Remesh to fine grid
    u0_coeffs_fine=remesh_1d(u0_coeffs, mesh_points, num_meshpoints, fine_mesh_points, num_fine_meshpoints, num_quad_points)

    # Time-stepping on fine grid
    with torch.no_grad():
        un_coeffs_fine = u0_coeffs_fine.clone().detach()
        un_coeffs_fine, _, sol_fine, BC1, BC2 = torch_FEM_Burgers_1D(opt, fine_mesh_points, quad_points, num_fine_meshpoints, un_coeffs_fine)

    # Time-stepping of on coarse grid
    un_coeffs1, _, sol, BC1, BC2 = torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints, u0_coeffs)

    #un_coeffs = un_coeffs1.clone().detach()

    #out, mesh_points_fem, sol, BC1, BC2 = torch_FEM_1D(opt, mesh_points, quad_points, num_meshpoints, c_list, s_list)

    # %% Compute Loss (L2 error)
    loss = F.mse_loss(sol, sol_fine)
    loss.backward()
    return torch.tensor(loss.detach().numpy()), mesh_points.grad

def build_mass_matrix(mesh_points, load_quad_points,num_meshpoints):
    M = torch.zeros(num_meshpoints, num_meshpoints)  # Mass matrix

    for n in range(num_meshpoints):
        def phi(x):
            return phim(x, mesh_points, n)

        M[:, n] = fast_inner_product(mesh_points, phi, load_quad_points,num_meshpoints)[:, 0]
    return M

def remesh_1d(un_coeffs,mesh_points, num_meshpoints, fine_mesh_points,num_fine_meshpoints, load_quad_points):
    def un(x):
        return fn_expansion(un_coeffs, mesh_points, x, num_meshpoints)

    return fast_inner_product(fine_mesh_points, un, load_quad_points,num_fine_meshpoints)[:, 0]

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

    #todo the solution calculation is not robust to negative mesh points - currently fixing with scaling and clipping
    sol = fn_expansion(unp1_coeffs, mesh_points, quad_points, num_meshpoints)

    return unp1_coeffs, mesh_points, sol, BC1, BC2

if __name__ == '__main__':
    # %% Parameters
    num_meshpoints = 50  # Means we have points x_n with n=0,1,...,N+1
    load_quad_points = 100  # Number of quadrature points for FEM assembly
    num_eval_quad_points = 100  # Number of quadrature points for evaluation
    tau = 0.01 #0.01#0.006 #2  # Gradient descent speed
    c_list = torch.tensor([np.array([0.5])])  # True value of c
    s_list = torch.tensor([np.array([0.1])])  # Scaling parameter
    mesh_points = torch.tensor(np.linspace(0, 1, num_meshpoints), requires_grad=True)
    quad_points = torch.tensor(np.linspace(0, 1, load_quad_points), requires_grad=False)
    nu=0.005 # Viscosity

    # Assembly mass matrix M and stiffness matrix A

    A=torch.zeros(num_meshpoints,num_meshpoints) # Stiffness matrx
    M=torch.zeros(num_meshpoints,num_meshpoints) # Mass matrix

    for n in range(num_meshpoints):
        def phi(x):
            return phim(x, mesh_points, n)

        M[:,n]=fast_inner_product(mesh_points, phi ,load_quad_points,num_meshpoints)[:,0]

    A=build_stiffness_matrix(mesh_points, quad_points, load_quad_points)

    c=3.0
    alpha=4.0
    # Initial condition
    def u0(x):
        #return 0.5 + 0.1 * torch.sin(2 * torch.pi * x)
        #return c-alpha*np.tanh(alpha*x/2.0/nu)
        #return -torch.sign(x-0.25)
        return torch.exp(-((x-c_list[0])**2)/s_list[0]**2)

    # Find expansion coefficients for the initial condition
    u0_coeffs = torch.tensor(np.linalg.solve(M.detach().numpy(),fast_inner_product(mesh_points, u0 ,load_quad_points,num_meshpoints)[:,0].detach().numpy()) )

    # Check with plot
    u0_expanded=fn_expansion(u0_coeffs, mesh_points, quad_points, num_eval_quad_points)
    dxu0_expanded=dxfn_expansion(u0_coeffs, mesh_points, quad_points, num_eval_quad_points)
    # Plotting of the initial condition
    x = torch.tensor(np.linspace(0, 1, num_eval_quad_points))
    plt.plot(x.detach().numpy(), u0(x).detach().numpy())
    plt.plot(x.detach().numpy(), u0_expanded.detach().numpy())
    #plt.plot(x, dxu0_expanded.detach().numpy())
    plt.title('Initial condition')
    plt.show()

    # # Time-stepping (heat equation to begin with
    # un_coeffs = u0_coeffs.clone().detach()
    # with torch.no_grad():
    #     for j in range(1000): # Careful with the minus sign because we integrate by parts!
    #         un_coeffs=un_coeffs.detach()-tau*nu*torch.linalg.solve(M,torch.matmul(A,un_coeffs))
    #         #torch.solve(torch.tensor([u0(mesh_points.numpy())]).T, M, out=(u0_coeffs, _))
    #         if j % 100 == 0:
    #             un_expanded=fn_expansion(un_coeffs, mesh_points, quad_points, num_eval_quad_points)
    #             plt.plot(x, un_expanded.detach().numpy())
    #             plt.title('Evolution')
    #             plt.show()

    # Transport part

    # def f(x):
    #     return fn_expansion(u0_coeffs, mesh_points, x, 100)*dxfn_expansion(u0_coeffs, mesh_points, x, 100)
    #
    # fast_inner_product(mesh_points, f ,load_quad_points)[:,0]
    # un_coeffs=u0_coeffs.clone().detach()
    # with torch.no_grad():
    #     for j in range(1000): # Careful with the minus sign because we integrate by parts!
    #
    #         un_coeffs=un_coeffs.detach()-tau*nu*torch.linalg.solve(M,torch.matmul(A,un_coeffs))
    #
    #
    #         def un_dun(x):
    #             return (fn_expansion(un_coeffs, mesh_points, x, num_meshpoints)) * dxfn_expansion(un_coeffs, mesh_points, x, num_meshpoints)
    #
    #         un_coeffs=un_coeffs-tau*fast_inner_product(mesh_points, un_dun, load_quad_points)[:, 0]
    #
    #         un_coeffs[0]=1.0
    #         un_coeffs[-1]=-1.0
    #         #torch.solve(torch.tensor([u0(mesh_points.numpy())]).T, M, out=(u0_coeffs, _))
    #         if j % 100 == 0:
    #             un_expanded=fn_expansion(un_coeffs, mesh_points, quad_points, num_eval_quad_points)
    #             plt.plot(x, un_expanded.detach().numpy())
    #             plt.title('Evolution transport')
    #     plt.show()
    #
    #     # Combined


    un_coeffs = u0_coeffs.clone().detach()
    # Boundary conditions
    BC1 = u0(torch.tensor([0.0]))  # 0.0 # Left endpoint Dirichlet BC
    BC2 = u0(torch.tensor([1.0]))

    with torch.no_grad():
        for j in range(1000):  # Careful with the minus sign because we integrate by parts!

            #un_coeffs = un_coeffs.detach() - tau * nu * torch.linalg.solve(M, torch.matmul(A, un_coeffs))


            def un_dun(x):
                return (fn_expansion(un_coeffs, mesh_points, x, num_meshpoints)) * dxfn_expansion(un_coeffs,
                                                                                                  mesh_points, x,
                                                                                                  num_meshpoints)

            RHS= torch.matmul(M, un_coeffs) - tau * fast_inner_product(mesh_points, un_dun, load_quad_points,num_meshpoints)[:, 0]

            # Correct for BCs

            RHS[0] = BC1
            RHS[-1] = BC2

            Matrix=M+tau*nu*A

            # Correct matrix for BCs
            Matrix[0,:] = torch.zeros(num_meshpoints)
            Matrix[-1,:] = torch.zeros(num_meshpoints)
            Matrix[0,0] = 1.0
            Matrix[-1,-1] = 1.0



            un_coeffs=torch.linalg.solve(Matrix,RHS)
            # torch.solve(torch.tensor([u0(mesh_points.numpy())]).T, M, out=(u0_coeffs, _))
            if j % 10 == 0:
                un_expanded = fn_expansion(un_coeffs, mesh_points, quad_points, num_eval_quad_points)
                plt.plot(x, un_expanded.detach().numpy())
                plt.title('Evolution full Burgers')
        plt.show()
    # Define opt as a dictionary
    opt = {}
    opt['load_quad_points']=100
    opt['stiff_quad_points']=100
    opt['eval_quad_points']=100
    opt['tau']=0.01
    opt['nu']=0.005

    un_coeffs = u0_coeffs.clone().detach()
    # Boundary conditions
    BC1 = u0(torch.tensor([0.0]))  # 0.0 # Left endpoint Dirichlet BC
    BC2 = u0(torch.tensor([1.0]))

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

    un_coeffs1, mesh_points, sol, BC1, BC2 = torch_FEM_Burgers_1D(opt, mesh_points, quad_points, num_meshpoints,
                                                                  un_coeffs, BC1, BC2)
    un_expanded = fn_expansion(un_coeffs1, mesh_points, quad_points, num_eval_quad_points)

    loss= torch.sum((u0_expanded - un_expanded) ** 2)

    # Test gradient computation
    x_phys=torch.tensor(np.linspace(0,1,num_meshpoints),requires_grad=True)
    data={}
    opt['mesh_dims']=[num_meshpoints]
    data['u0']=u0
    a,b=gradient_meshpoints_1D_Burgers_PDE_loss_direct_mse(opt,data, x_phys)