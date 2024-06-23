import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Define error measure (here we have m(x)=(1+|u_xx|^2)^(1/5)
def m(x, params):
    c_list = params['centers']
    s_list = params['scales']
    if 'mon_reg' in params.keys() and 'mon_power' in params.keys():
        return (params['mon_reg'] + diag_hessian(x, c_list, s_list))**params['mon_power']
    elif 'mon_power' in params.keys():
        return (1 + diag_hessian(x, c_list, s_list))**params['mon_power'] #default 0.2
    else:
        return (1 + diag_hessian(x, c_list, s_list))**0.2#0.05

def diag_hessian_old(x, c_list=[[0.25,0.25]], s_list=[[0.2,0.2]]): # outputs sqrt(|u_xx|^2+|u_yy|^2) for the exact Gaussian
    u_xx = torch.zeros_like(x)
    for i in range(len(c_list)):
        c = c_list[i] # center of Gaussian reference solution
        s = s_list[i]# scaling of Gaussian reference solution

        u_xx += torch.abs(-((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4)*torch.exp(-(x-c[0])**2/s[0]**2)**2)
    return u_xx
def diag_hessian(x, c_list=[[0.25,0.25]], s_list=[[0.2,0.2]]): # outputs sqrt(|u_xx|^2+|u_yy|^2) for the exact Gaussian
    u_xx = torch.zeros_like(x)
    for i in range(len(c_list)):
        c = c_list[i] # center of Gaussian reference solution
        s = s_list[i]# scaling of Gaussian reference solution

        #u_xx += torch.abs(-((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4)*torch.exp(-(x-c[0])**2/s[0]**2)**2)
        u_xx += -((2 * (-2 * c[0] ** 2 + s[0] ** 2 + 4 * c[0] * x - 2 * x ** 2)) / s[0] ** 4)*torch.exp(-(x-c[0])**2/s[0]**2)
    return (u_xx)**2/torch.max((u_xx)**2) # Normalize to 1


# Define right hand side with finite differences in MMPDE5
def RHS(m, X, N, params):
    Xix = torch.linspace(0, 1, N) # Define mesh in computational domain
    Xixfine = torch.linspace(0, 1, 2 * N - 1) #MMPDE5 half grid size
    deltaXi=Xix[1] # Spatial grid size (in Xi domain)
    tau= 0.1#0.02#0.1 # Parameter in MMPDE5

    mvec = m(Xixfine, params)
    mvec2 = m(Xix, params)
    mvecshort = mvec[1:2 * N - 1:2]

    # Output approximation to d/dx (m d/dx(u))
    return (mvecshort[1:N-1]*(X[2:N]-X[1:N-1])-mvecshort[0:N-2]*(X[1:N-1]-X[0:N-2]))/deltaXi**2/tau/mvec2[1:N-1]

def RHS_burgers(m, X, N):
    Xix = torch.linspace(0, 1, N) # Define mesh in computational domain
    Xixfine = torch.linspace(0, 1, 2 * N - 1) #MMPDE5 half grid size
    deltaXi=Xix[1] # Spatial grid size (in Xi domain)
    tau= 0.1#0.02#0.1 # Parameter in MMPDE5

    mvec = m(Xixfine)
    mvec2 = m(Xix)
    mvecshort = mvec[1:2 * N - 1:2]

    # Output approximation to d/dx (m d/dx(u))
    return (mvecshort[1:N-1]*(X[2:N]-X[1:N-1])-mvecshort[0:N-2]*(X[1:N-1]-X[0:N-2]))/deltaXi**2/tau/mvec2[1:N-1]


# Define generic RK4 time-stepping
def RK4(x,f,h):
    k1=f(x)
    k2=f(x+h*k1/2)
    k3=f(x+h*k2/2)
    k4=f(x+h*k3)
    return x+h/6*(k1+2*k2+2*k3+k4)


def f(X, N, params):
    A = torch.zeros( X.shape[0])
    sol = RHS(m, X, N, params)  # Perform time-stepping using the above RHS to move the mesh
    # A[:, 1:N - 1, 1:N - 1] = sol # Zero padding to fix boundary
    A[ 1:N-1] = sol # Zero padding to fix boundary
    return A

def f_burgers(m, X, N):
    A = torch.zeros( X.shape[0])
    sol = RHS_burgers(m, X, N)  # Perform time-stepping using the above RHS to move the mesh
    # A[:, 1:N - 1, 1:N - 1] = sol # Zero padding to fix boundary
    A[ 1:N-1] = sol # Zero padding to fix boundary
    return A


def MMPDE5_1d(X, N, params):
    convergence_measure = 1.0  # Measures change in each RK4 timestep to serve as stopping criterion
    j = 0  # number of time steps
    CFL = 0.05 # For MMPDE5 solver: Need to restrict time step since solve with explicit RK4
    tol = 1e-6 # For MMPDE5 solver: tolerance for stopping criterion

    start_time = time.time()
    while j < 10000 and convergence_measure > tol:
        j = j + 1
        Xold = X.clone() # Save previous values
        # [X, Y] = RK4([X, Y], f, CFL / N ** 3)  # Single time step in MMPDE5
        X = RK4(X, lambda x: f(x, N, params), CFL / N ** 3)  #need partial f to pass N
        convergence_measure = (torch.sum(np.abs(X - Xold)))  # Measure size of update
        if convergence_measure > 1.0 / tol:  # CFL needs to be adapted if the PDE is very stiff (large m(x))
            print('Warning: MMPDE5 is too stiff, please choose smaller CFL')
            break
        # print(convergence_measure)
    build_time = time.time() - start_time

    if convergence_measure > tol:
        print('Warning: MMPDE5 has not yet converged to stationary solution.')

    return X, j, build_time

def MMPDE5_1d_burgers(m, X, N):
    convergence_measure = 1.0  # Measures change in each RK4 timestep to serve as stopping criterion
    j = 0  # number of time steps
    CFL = 0.05 # For MMPDE5 solver: Need to restrict time step since solve with explicit RK4
    tol = 1e-6 # For MMPDE5 solver: tolerance for stopping criterion

    start_time = time.time()
    while j < 10000 and convergence_measure > tol:
        j = j + 1
        Xold = X.clone() # Save previous values
        # [X, Y] = RK4([X, Y], f, CFL / N ** 3)  # Single time step in MMPDE5
        X = RK4(X, lambda x: f_burgers(m, x, N), CFL / N ** 3)  #need partial f to pass N
        convergence_measure = (torch.sum(np.abs(X - Xold)))  # Measure size of update
        if convergence_measure > 1.0 / tol:  # CFL needs to be adapted if the PDE is very stiff (large m(x))
            print('Warning: MMPDE5 is too stiff, please choose smaller CFL')
            break
        # print(convergence_measure)
    build_time = time.time() - start_time

    if convergence_measure > tol:
        print('Warning: MMPDE5 has not yet converged to stationary solution.')

    return X, j, build_time


if __name__ == "__main__":
    print("test")
    N=20
    params={'centers':torch.tensor([[0.5]]),'scales':torch.tensor([[0.1]])}
    c=params['centers']
    print(c[0])
    x0=torch.tensor(np.linspace(0.0,1.0,num=N))
    X=MMPDE5_1d(x0, N, params)
    plt.plot(1.0 * np.ones(N), X[0].numpy(), '*')
    # plt.show()
