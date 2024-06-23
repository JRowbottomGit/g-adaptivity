import torch
import numpy as np
import matplotlib.pyplot as plt
# from torchquad import Trapezoid
import pykeops
from pykeops.torch import LazyTensor

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

def soln(x, out, mesh, BC1, BC2, num_meshpoints): # Compute solution values of our approximation
    #todo speed up with matrix multiplication and vectorisation
    interior=0.0*x
    for m in range(num_meshpoints):
        interior=interior+out[m]*phim(x,mesh,m+1)
    return interior+BC1*phim(x,mesh,0)+BC2*phim(x,mesh,num_meshpoints+1)

#%% L2 error
def L2norm(u,x): # Compute L2 norm of the function values u.
    return torch.trapezoid(abs(u)**2,x)

def build_stiffness_matrix(mesh, x, num_meshpoints, vectorise=True, keyops=False):
    if keyops:
        pass
        #todo vectorise basis functions and add pykeops

    elif vectorise:
        #FOR OFF-DIAGONAL ELEMENTS
        #1) define quadrature points in overlapping support of basis functions
        k = 10
        diffs = torch.diff(mesh) / k
        #check any negative diffs
        if torch.any(diffs < 0.):
            print("WARNING: negative diffs in build_mass_matrix")
        L_start = mesh[:-1].view(-1, 1)
        x_diffs = torch.arange(k + 1).view(1, -1).repeat(L_start.shape[0], 1) * diffs.view(-1, 1)
        x_off_diag = L_start + x_diffs

        #2) compute product of derivatives of basis functions at quadrature points
        a = mesh[:-1]
        b = mesh[1:]
        L_dphi = (-1 / (b - a)).view(-1, 1).repeat(1, k + 1)
        R_dphi = (1 / (b - a)).view(-1, 1).repeat(1, k + 1)

        # 3) compute mass matrix
        off_diags = torch.trapezoid(L_dphi * R_dphi, x_off_diag)

        # FOR DIAGONAL ELEMENTS
        x_diag = torch.cat((x_off_diag[:-1], x_off_diag[1:,1:]), dim=1)
        values_diag = torch.cat((R_dphi[:-1], L_dphi[1:,1:]), dim=1)
        internal_diag = torch.trapezoid(values_diag**2, x_diag)

        #FOR LEFT AND RIGHT BOUNDARY ELEMENTS #not needed for zero BCs
        A = torch.zeros(num_meshpoints, num_meshpoints)
        A[:,:] = -torch.diag(internal_diag)
        A[:-1, 1:] -= torch.diag(off_diags[1:-1])
        A[1:, :-1] -= torch.diag(off_diags[1:-1])

        return A

    else:
        A = torch.zeros(num_meshpoints+2, num_meshpoints+2)  # Stiffness matrx
        for m in range(num_meshpoints+2):
            for n in range(num_meshpoints+2):
                A[m, n] = -torch.trapezoid(dphim(x, mesh, m) * dphim(x, mesh, n), x)

    return A

def build_mass_matrix(mesh, x, num_meshpoints, vectorise=True, keyops=False):
    if keyops:
        pass

    else:
        M = torch.zeros(num_meshpoints+2, num_meshpoints+2)  # Mass matrx
        for m in range(num_meshpoints+2):
            for n in range(num_meshpoints+2):
                M[m, n] = torch.trapezoid(phim(x, mesh, m) * phim(x, mesh, n), x)

    return M

def build_load_vector(mesh, x, BC1, BC2, num_meshpoints,s=0.1):
    RHS = torch.zeros(num_meshpoints, 1)  # Forcing
    for m in range(num_meshpoints):
        RHS[m] = torch.trapezoid(phim(x, mesh, m + 1) * f(x, s), x) + BC1 * torch.trapezoid(
            dphim(x, mesh, m + 1) * dphim(x, mesh, 0), x) + BC2 * torch.trapezoid(
            dphim(x, mesh, m + 1) * dphim(x, mesh, num_meshpoints + 1), x)

    return RHS


def u0(x): # Initial condition
    return torch.cos(2*torch.pi*x)

def RHS_initial_coeff(mesh, x, u0, num_meshpoints):
    RHS = torch.zeros(num_meshpoints+2, 1)  # Forcing
    for m in range(num_meshpoints+2):
        RHS[m] = torch.trapezoid(phim(x, mesh, m ) * u0(x), x)
    return RHS

def Burgers_nonlinearity(mesh, x, a, num_meshpoints):
    output = torch.zeros(num_meshpoints+2, 1)  # Forcing
    for m in range(num_meshpoints+2):
        for n in range(num_meshpoints+2):
            for l in range(num_meshpoints+2):
                output[m] = output[m]+torch.trapezoid(phim(x, mesh, m ) * a[n]*phim(x, mesh, n )*a[l]*dphim(x,mesh,l), x)
    return output


if __name__ == '__main__':
    # %% Parameters
    num_meshpoints = 20#1  # Means we have points x_n with n=0,1,...,N+1
    num_quad_points = 2000  # Number of quadrature points for FEM assembly
    tau = 0.006 #0.01#0.006 #2  # Gradient descent speed
    s = 0.1  # Scaling parameter
    dt=0.01 # Time step in Burgers equation
    nu=0.02

    # Below can be outsourced
    mesh = torch.tensor(np.linspace(0, 1, num_meshpoints + 2), requires_grad=True)

    # %% Assembly of linear system
    x = torch.tensor(np.linspace(0, 1, num_quad_points))  # Define quadrature points

    # Assemble matrices
    A=build_stiffness_matrix(mesh, x, num_meshpoints, vectorise=False, keyops=False)
    M=build_mass_matrix(mesh, x, num_meshpoints, vectorise=False, keyops=False)

    # Compute RHS
    u0_RHS=RHS_initial_coeff(mesh, x, u0, num_meshpoints)

    a0=torch.linalg.solve(M,u0_RHS)

    # IMEX time stepping
    a=a0

    for j in range(40):
        with torch.no_grad():
            a=torch.linalg.solve(M-dt*nu*A,torch.matmul(M,a)-dt*Burgers_nonlinearity(mesh, x, a, num_meshpoints))

        if j%5==0:
            plt.plot(mesh.detach().numpy(),a.detach().numpy())
            plt.show()




