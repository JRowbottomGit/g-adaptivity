import time
import numpy as np
import torch
from firedrake import UnitSquareMesh, triplot, SpatialCoordinate, exp, sqrt, Constant, FunctionSpace, Function, div, grad
from movement import *
from firedrake_difFEM.solve_poisson import poisson2d_fmultigauss_bcs,poisson2d_fmultigauss_bcs_high_order
from src.utils_eval import firedrake_call_fct_handler


# Define error measure (here we have m(x)=(1+|u_xx|^2+|u_yy|^2)^(1/5)
def m(x, y, params):
    c_list = params['centers']
    s_list = params['scales']
    if 'mon_power' in params.keys():
        return (1 + diag_hessian(x, y, c_list, s_list))**params['mon_power'] #default 0.2
    else:
        return (1 + diag_hessian(x, y, c_list, s_list))**0.2#0.05



def diag_hessian(x, y, c_list=[[0.25,0.25]], s_list=[[0.2,0.2]]): # outputs sqrt(|u_xx|^2+|u_yy|^2) for the exact Gaussian
    u_xx = torch.zeros_like(x)
    u_yy = torch.zeros_like(x)

    for i in range(len(c_list)):
        c = c_list[i] # center of Gaussian reference solution
        s = s_list[i]# scaling of Gaussian reference solution

        u_xx += torch.abs(-((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4)*torch.exp(-(x-c[0])**2/s[0]**2-(y-c[1])**2/s[1]**2))
        u_yy += torch.abs(-((2 * (-2 * c[1] ** 2 + s[1] ** 2 + 4 * c[1] * y - 2 * y ** 2)) / s[1] ** 4)*torch.exp(-(x-c[0])**2/s[0]**2-(y-c[1])**2/s[1]**2))

    return u_xx**2 + u_yy**2


# Define right hand side with finite differences in MMPDE5
def RHS(m, X, Y, N, params):
    Xix, Xiy = torch.meshgrid(torch.linspace(0, 1, N), torch.linspace(0, 1, N)) # Define mesh in computational domain
    Xixfine, Xiyfine = torch.meshgrid(torch.linspace(0, 1, 2 * N - 1), torch.linspace(0, 1, 2 * N - 1)) #MMPDE5 half grid size
    deltaXi=Xix[1,0] # Spatial grid size (in Xi domain)
    tau= 0.1#0.02#0.1 # "speed" parameter in MMPDE5

    mvec = m(Xixfine, Xiyfine, params)
    mvec2 = m(Xix, Xiy, params)
    mvecshort = mvec[1:2 * N - 1:2, 1:2 * N - 1:2]

    # Construct four auxiliary components corresponding to \nabla\cdot(m\nabla(u))
    aux1 = (mvecshort[1:N - 1, 1:N - 1] * (X[2:N, 1:N - 1] - X[1:N - 1, 1:N - 1]) - mvecshort[0:N - 2, 1:N - 1] * (
                 X[1:N - 1, 1:N - 1] - X[0:N - 2, 1:N - 1])) / deltaXi ** 2 / tau / mvec2[1:N-1,1:N-1]
    aux2 = (mvecshort[1:N - 1, 1:N - 1] * (X[1:N - 1, 2:N] - X[1:N - 1, 1:N - 1]) - mvecshort[1:N - 1, 0:N - 2] * (
                 X[1:N - 1, 1:N - 1] - X[1:N - 1, 0:N - 2])) / deltaXi ** 2 / tau / mvec2[1:N-1,1:N-1]

    aux3 = (mvecshort[1:N - 1, 1:N - 1] * (Y[2:N, 1:N - 1] - Y[1:N - 1, 1:N - 1]) - mvecshort[0:N - 2, 1:N - 1] * (
            Y[1:N - 1, 1:N - 1] - Y[0:N - 2
    , 1:N - 1])) / deltaXi ** 2 / tau / mvec2[1:N-1,1:N-1]
    aux4 = (mvecshort[1:N - 1, 1:N - 1] * (Y[1:N - 1, 2:N] - Y[1:N - 1, 1:N - 1]) - mvecshort[1:N - 1, 0:N - 2] * (
            Y[1:N - 1, 1:N - 1] - Y[1:N - 1, 0:N - 2])) / deltaXi ** 2 / tau / mvec2[1:N-1,1:N-1]

    return [aux1 + aux2, aux3 + aux4]
#for debugging
#mvecshort[1:N - 1, 1:N - 1], X[2:N, 1:N - 1], X[1:N - 1, 1:N - 1], mvecshort[0:N - 2, 1:N - 1], X[1:N - 1, 1:N - 1], X[0:N - 2, 1:N - 1],mvecshort[1:N - 1, 1:N - 1], X[1:N - 1, 2:N], X[1:N - 1, 1:N - 1], mvecshort[1:N - 1, 0:N - 2], X[1:N - 1, 1:N - 1], X[1:N - 1, 0:N - 2]

# Define generic RK4 time-stepping
def RK4(x, f, h):
    x = torch.stack(x)
    k1=f(x)
    k2=f(x+h*k1/2)
    k3=f(x+h*k2/2)
    k4=f(x+h*k3)
    return x+h/6*(k1+2*k2+2*k3+k4)


def f(XY, N, params):
    A = torch.zeros(2, XY[0].shape[0], XY[0].shape[1])
    sol = RHS(m, XY[0], XY[1], N, params)  # Perform time-stepping using the above RHS to move the mesh
    sol = torch.stack(sol)
    # A[:, 1:N - 1, 1:N - 1] = sol # Zero padding to fix boundary
    A[:, 1:N-1, 1:N-1] = sol # Zero padding to fix boundary
    return A


def MMPDE5_2d(X, Y, N, params):
    convergence_measure = 1.0  # Measures change in each RK4 timestep to serve as stopping criterion
    j = 0  # number of time steps
    CFL = 0.05 # For MMPDE5 solver: Need to restrict time step since solve with explicit RK4
    tol = 1e-6 # For MMPDE5 solver: tolerance for stopping criterion

    start_time = time.time()
    while j < 10000 and convergence_measure > tol:
        j = j + 1
        [Xold, Yold] = [X.clone(), Y.clone()]  # Save previous values
        # [X, Y] = RK4([X, Y], f, CFL / N ** 3)  # Single time step in MMPDE5
        [X, Y] = RK4([X, Y], lambda x: f(x, N, params), CFL / N ** 3)  #need partial f to pass N
        convergence_measure = (torch.sum(np.abs(X - Xold) + torch.abs(Y - Yold)))  # Measure size of update
        if convergence_measure > 1.0 / tol:  # CFL needs to be adapted if the PDE is very stiff (large m(x))
            print('Warning: MMPDE5 is too stiff, please choose smaller CFL')
            break
        # print(convergence_measure)
    build_time = time.time() - start_time

    if convergence_measure > tol:
        print('Warning: MMPDE5 has not yet converged to stationary solution.')

    return X, Y, j, build_time


def diag_hessian_ma(x,y, c_list=[[0.25,0.25]], s_list=[[0.2,0.2]]): # outputs sqrt(|u_xx|^2+|u_yy|^2) for the exact Gaussian
    u_xx=0*x
    u_yy=0*x

    for i in range(len(c_list)):
        # c = c_list[i] # center of Gaussian reference solution
        # s = s_list[i]# scaling of Gaussian reference solution
        c = Constant(c_list[i])
        s = Constant(s_list[i])
        u_xx += (-((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4)*exp(-(x-c[0])**2/s[0]**2-(y-c[1])**2/s[1]**2))
        u_yy += (-((2 * (-2 * c[1] ** 2 + s[1] ** 2 + 4 * c[1] * y - 2 * y ** 2)) / s[1] ** 4)*exp(-(x-c[0])**2/s[0]**2-(y-c[1])**2/s[1]**2))
    return sqrt(u_xx**2+u_yy**2)

def froboenius_norm_hessian_ma(x,y, c_list=[[0.25,0.25]], s_list=[[0.2,0.2]]): # outputs sqrt(|u_xx|^2+4|u_xy|^2+|u_yy|^2) for the exact Gaussian
    u_xx = 0 * x
    u_yy = 0 * x
    u_xy = 0 * x
    for i in range(len(c_list)):
        # c = c_list[i] # center of Gaussian reference solution
        # s = s_list[i]# scaling of Gaussian reference solution
        c = Constant(c_list[i])
        s = Constant(s_list[i])
        u_xx += (-((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4) * exp(
            -(x - c[0]) ** 2 / s[0] ** 2 - (y - c[1]) ** 2 / s[1] ** 2))
        u_yy += (-((2 * (-2 * c[1] ** 2 + s[1] ** 2 + 4 * c[1] * y - 2 * y ** 2)) / s[1] ** 4) * exp(
            -(x - c[0]) ** 2 / s[0] ** 2 - (y - c[1]) ** 2 / s[1] ** 2))

        u_xy = ((16 * ((x-c[0])*(y-c[1])) / s[0] ** 2 / s[1] **2) * exp(-(x - c[0]) ** 2 / s[0] ** 2 - (y - c[1]) ** 2 / s[1] ** 2))

    return sqrt(u_xx**2 + 2 * u_xy**2 + u_yy**2)

def froboenius_norm_hessian_np(x,y, c_list=[[0.25,0.25]], s_list=[[0.2,0.2]]): # outputs sqrt(|u_xx|^2+4|u_xy|^2+|u_yy|^2) for the exact Gaussian
    u_xx = 0 * x
    u_yy = 0 * x
    u_xy = 0 * x
    for i in range(len(c_list)):
        # c = c_list[i] # center of Gaussian reference solution
        # s = s_list[i]# scaling of Gaussian reference solution
        c = c_list[i]
        s = s_list[i]
        u_xx += (-((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4) * np.exp(
            -(x - c[0]) ** 2 / s[0] ** 2 - (y - c[1]) ** 2 / s[1] ** 2))
        u_yy += (-((2 * (-2 * c[1] ** 2 + s[1] ** 2 + 4 * c[1] * y - 2 * y ** 2)) / s[1] ** 4) * np.exp(
            -(x - c[0]) ** 2 / s[0] ** 2 - (y - c[1]) ** 2 / s[1] ** 2))

        u_xy = ((16 * ((x-c[0])*(y-c[1])) / s[0] ** 2 / s[1] **2) * np.exp(-(x - c[0]) ** 2 / s[0] ** 2 - (y - c[1]) ** 2 / s[1] ** 2))

    return np.sqrt(u_xx**2 + 2 * u_xy**2 + u_yy**2)
def m(x, y, params):
    c_list = params['centers']
    s_list = params['scales']
    if 'mon_power' in params.keys():
        return (1 + diag_hessian(x, y, c_list, s_list))**params['mon_power'] #default 0.2
    else:
        return (1 + diag_hessian(x, y, c_list, s_list))**0.2#0.05


def MA2d(x_comp, N, params):
    mesh = UnitSquareMesh(N-1, N-1)

    mesh.coordinates.dat.data[:, :]=x_comp

    def monitor(mesh):
        c_list = params['centers']
        s_list = params['scales']
        x, y = SpatialCoordinate(mesh)
        if params['mesh_type'] is 'ma':
            if 'mon_power' in params.keys():  # check if mon_power is float
                return (params['mon_reg'] + diag_hessian_ma(x, y, c_list, s_list)) ** params['mon_power']  # default 0.2
            else:
                return (1.0 + diag_hessian_ma(x, y, c_list, s_list)) ** 0.2

        elif params['mesh_type'] is 'M2N':
            # M2N monitor
            V = FunctionSpace(mesh, "CG", 1)  # Auxiliary function space required for MA mover when referring to approx solution

            if 'fast_M2N_monitor' in params.keys() and params['fast_M2N_monitor']=='superslow':

                if 'M2N_alpha' in params.keys():
                    alpha = params['M2N_alpha']
                else:
                    alpha = 1.0
                if 'M2N_beta' in params.keys():
                    beta = params['M2N_beta']
                else:
                    beta = 1.0

                # M2N monitor with direct computation of Hessian from approximate solution
                uu, u_true, _ = poisson2d_fmultigauss_bcs_high_order(mesh, c_list, s_list)

                u_x = Function(V)
                u_y = Function(V)
                u_xx = Function(V)
                u_xy = Function(V)
                u_yy = Function(V)
                u_x.interpolate(grad(uu)[0])
                u_y.interpolate(grad(uu)[1])

                u_xx.interpolate(grad(u_x)[0])
                u_yy.interpolate(grad(u_y)[1])
                u_xy.interpolate(grad(u_x)[1])

                Hessian_term = Function(V)
                Hessian_term.interpolate(sqrt(u_xx**2 + 2 * u_xy**2 + u_yy**2))

                mesh_points = mesh.coordinates.dat.data_ro
                maximumHessian_term = np.max(np.abs(Hessian_term(mesh_points.tolist()))) # For rescaling Frobenius term

                # |u-u_true|^2 term
                square_diff = Function(V)
                square_diff.interpolate((uu - u_true) ** 2)
                uu_eval = firedrake_call_fct_handler(fd_fct=uu, point_list=mesh.coordinates.dat.data_ro.tolist())[0]
                u_true_eval = \
                firedrake_call_fct_handler(fd_fct=u_true, point_list=mesh.coordinates.dat.data_ro.tolist())[0]

                max_square_diff = np.max((uu_eval - u_true_eval) ** 2)  # For rescaling square_diff term

                output = Function(V)  # Output, need to reinterpolate the monitor fn to mesh function space
                output.interpolate(square_diff)
                output.interpolate(Constant(params['mon_reg']) + alpha * square_diff / max_square_diff + beta * Hessian_term / maximumHessian_term)

            elif 'fast_M2N_monitor' in params.keys() and params['fast_M2N_monitor'] == 'slow':

                if 'M2N_alpha' in params.keys():
                    alpha = params['M2N_alpha']
                else:
                    alpha = 1.0
                if 'M2N_beta' in params.keys():
                    beta = params['M2N_beta']
                else:
                    beta = 1.0

                # alpha = 1.0
                # beta = params['M2N_beta']

                # M2N monitor with analytical computation of Hessian term
                uu, u_true, _ = poisson2d_fmultigauss_bcs(mesh, c_list, s_list, fast=True)

                Hessian_term = Function(V)

                Hessian_term.interpolate(froboenius_norm_hessian_ma(x, y, c_list, s_list))
                mesh_points = mesh.coordinates.dat.data_ro
                maximumHessian_term = np.max(froboenius_norm_hessian_np(mesh_points[:, 0], mesh_points[:, 1], c_list, s_list))  # For rescaling Frobenius term

                # |u-u_true|^2 term
                square_diff = Function(V)
                square_diff.interpolate((uu - u_true) ** 2)
                uu_eval = firedrake_call_fct_handler(fd_fct=uu, point_list=mesh.coordinates.dat.data_ro.tolist())[0]
                u_true_eval = \
                firedrake_call_fct_handler(fd_fct=u_true, point_list=mesh.coordinates.dat.data_ro.tolist())[0]

                max_square_diff = np.max((uu_eval - u_true_eval) ** 2)  # For rescaling square_diff term

                output = Function(V)  # Output, need to reinterpolate the monitor fn to mesh function space
                output.interpolate(square_diff)
                output.interpolate(Constant(params['mon_reg']) + alpha * square_diff / max_square_diff + beta * Hessian_term / maximumHessian_term)

            elif 'fast_M2N_monitor' in params.keys() and params['fast_M2N_monitor'] == 'fast':
                # beta=1.5
                if 'M2N_beta' in params.keys():
                    beta = params['M2N_beta']
                else:
                    beta = 1.5

                mesh_points = mesh.coordinates.dat.data_ro
                maximumHessian_term = np.max(froboenius_norm_hessian_np(mesh_points[:, 0], mesh_points[:, 1], c_list, s_list))  # For rescaling Frobenius term
                output = params['mon_reg'] + beta*froboenius_norm_hessian_ma(x, y, c_list, s_list) / maximumHessian_term


            else:
                raise ValueError('Please specify fast_M2N_monitor in params')

            return output


    mover = MongeAmpereMover(mesh, monitor, method="quasi_newton")

    start_time = time.time()

    try:
        j = mover.move()
        # x_phys= mesh.coordinates.dat.data[:, :]
        x_phys = mover.mesh.coordinates.dat.data[:, :]
    except:
        j = 0
        x_phys = np.zeros_like(mesh.coordinates.dat.data[:, :])

    build_time = time.time() - start_time

    return x_phys, j, build_time


if __name__ == "__main__":
    pass