# Poisson equation
# ================

from firedrake import *
from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt

def poisson2d_fgauss_b0(mesh, c=[0.5, 0.5], s=[0.2, 0.2]):
    '''solves Poisson's equation on a given mesh with f gaussian and b=0'''
    pde_params = {'centers': [c], 'scales': [s]}

    #%% Define function spaces and PDE
    V = FunctionSpace(mesh, "CG", 1) # Piecewise linear splines

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx # weak form of Laplacian

    #%% Define forcing/right hand side term based on family of exact solutions
    x = SpatialCoordinate(mesh)
    u_true = Function(V) # True solution (Gaussian with extra factor to ensure zero Dirichlet BC)
    u_true.interpolate(exp(-(x[0]-c[0])**2/s[0]**2-(x[1]-c[1])**2/s[1]**2))

    F = Function(V) # Forcing (=-2*Laplacian of u_true)
    F.interpolate(-(1 / (s[0]**4 * s[1]**4)) * exp(-((c[0] - x[0])**2 / s[0]**2) - (c[1] - x[1])**2 / s[1]**2) *
                  (4 * c[1]**2 * s[0]**4 - 2 * s[0]**2 * s[1]**4 + 4 * s[1]**4 * (c[0] - x[0])**2 - 8 * c[1] * s[0]**4 * x[1] - 2 * s[0]**4 * (s[1]**2 - 2 * x[1]**2)))

    L = F*v*dx # Define linear form in weak formulation of the Poisson problem

    #%% Define boundary conditions
    bc = DirichletBC(V, 0, "on_boundary")
    boundary_nodes = bc.nodes
    bcs = [bc]

    #%% Initialise the approximate solution and solve with preconditioning
    uu = Function(V)
    uu.assign(0)

    solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    plot_solutions(uu, u_true)

    #%% Evaluate the L2 error
    print('\n\nFinished solve, L^2 error is:')
    print(sqrt(assemble(inner(uu - u_true, uu - u_true) * dx)))

    return uu, u_true, F, pde_params

def poisson1d_fmultigauss_bcs(mesh, c_list, s_list, num_gaussians=1, rand_gaussians=False, bc_type="u_true"):
    '''solves Poisson's equation on a given mesh with f MULTI gaussian and b=0
    rand_gaussians: if True, sample num_gaussians centers and scales from a uniform distribution
    :param bc_type: '''

    V = FunctionSpace(mesh, "CG", 1) # Piecewise linear splines

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx # weak form of Laplacian

    x = SpatialCoordinate(mesh)

    # Initialize the true solution and forcing term
    u_true_total = 0.#Constant(0, domain=mesh) #0
    F_total = 0.#Constant(0, domain=mesh) #0

    if rand_gaussians:
        num_gauss = num_gaussians
    else:
        num_gauss = len(c_list)

    for i in range(num_gauss):
        c = Constant(c_list[i])  # Ensure c is a Constant
        s = Constant(s_list[i])  # Ensure s is a Constant

        # Compute the Gaussian and its Laplacian
        gaussian = exp(-(x[0]-c[0])**2/s[0]**2)
        laplacian = (-2 / s[0]**2 + 4 * (x[0] - c[0])**2 / s[0]**4) * exp(-(x[0] - c[0])**2 / s[0]**2)
        # Compute exact gradient
        grad_gaussian_x = -2 * (x[0] - c[0]) / s[0] ** 2 * gaussian
        # Compute exact Hessian
        hessian_gaussian_xx = (4 * (x[0] - c[0]) ** 2 / s[0] ** 4 - 2 / s[0] ** 2) * gaussian

        # Add the Gaussian and its Laplacian to the total solution and forcing term
        u_true_total += gaussian
        F_total += laplacian

    # Interpolate the total solution and forcing term
    #u_true.interpolate(u_true_total)
    #F.interpolate(F_total)
    
    # Project the forcing function into the function space
    u_true = project(u_true_total, V, name="u_true")
    F = project(F_total, V, name="f")

    # Define boundary conditions
    if bc_type == "u_true":
        bc = DirichletBC(V, u_true, "on_boundary")
    elif bc_type == "zero":
        bc = DirichletBC(V, 0, "on_boundary")
    bcs = [bc]

    # Initialise the approximate solution and solve with preconditioning
    uu = Function(V, name="uu")
    uu.assign(0)

    #note compared to 2D notation we use -F here just because of the way we defined the Laplacian
    solve(a == -F * v * dx, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    #%% Evaluate the L2 error
    print('\n\nFinished solve, L^2 error is:')
    print(sqrt(assemble(inner(uu - u_true, uu - u_true) * dx)))

    return uu, u_true, F #, pde_params


def poisson2d_fmultigauss_bcs(mesh, c_list, s_list, num_gaussians=1, rand_gaussians=False, bc_type="u_true", fast=False):
    '''solves Poisson's equation on a given mesh with f MULTI gaussian and b=0
    rand_gaussians: if True, sample num_gaussians centers and scales from a uniform distribution
    :param bc_type: '''

    V = FunctionSpace(mesh, "CG", 1) # Piecewise linear splines

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx # weak form of Laplacian
    x = SpatialCoordinate(mesh)

    # Initialize the true solution and forcing term
    u_true_total = 0.
    F_total = 0.

    if rand_gaussians:
        num_gauss = num_gaussians
    else:
        num_gauss = len(c_list)

    for i in range(num_gauss):
        c = Constant(c_list[i])  # Ensure c is a Constant
        s = Constant(s_list[i])  # Ensure s is a Constant

        # Compute the Gaussian and its Laplacian
        gaussian = exp(-(x[0]-c[0])**2/s[0]**2-(x[1]-c[1])**2/s[1]**2)
        laplacian = -(1 / (s[0]**4 * s[1]**4)) * exp(-((c[0] - x[0])**2 / s[0]**2) - (c[1] - x[1])**2 / s[1]**2) \
                    * (4 * c[1]**2 * s[0]**4 - 2 * s[0]**2 * s[1]**4 + 4 * s[1]**4 * (c[0] - x[0])**2 - 8 * c[1] * s[0]**4 * x[1] - 2 * s[0]**4 * (s[1]**2 - 2 * x[1]**2))

        # Compute exact gradient
        grad_gaussian_x = -2 * (x[0] - c[0]) / s[0] ** 2 * gaussian
        grad_gaussian_y = -2 * (x[1] - c[1]) / s[1] ** 2 * gaussian
        # Compute exact Hessian
        hessian_gaussian_xx = (4 * (x[0] - c[0]) ** 2 / s[0] ** 4 - 2 / s[0] ** 2) * gaussian
        hessian_gaussian_yy = (4 * (x[1] - c[1]) ** 2 / s[1] ** 4 - 2 / s[1] ** 2) * gaussian
        hessian_gaussian_xy = 2 * (x[0] - c[0]) * (x[1] - c[1]) / (s[0] ** 2 * s[1] ** 2) * gaussian

        # Add the Gaussian and its Laplacian to the total solution and forcing term
        u_true_total += gaussian
        F_total += laplacian

    # Interpolate the total solution and forcing term
    #u_true.interpolate(u_true_total)
    #F.interpolate(F_total)

    # Project the forcing function into the function space
    u_true = project(u_true_total, V, name="u_true")
    F = project(F_total, V, name="f")

    # Define boundary conditions
    if bc_type == "u_true":
        bc = DirichletBC(V, u_true, "on_boundary")
    elif bc_type == "zero":
        bc = DirichletBC(V, 0, "on_boundary")
    bcs = [bc]

    # Initialise the approximate solution and solve with preconditioning
    uu = Function(V, name="uu")
    uu.assign(0)

    solve(a == F * v * dx, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    if fast == False:
        #%% Evaluate the L2 error
        print('\n\nFinished solve, L^2 error is:')
        print(sqrt(assemble(inner(uu - u_true, uu - u_true) * dx)))

    return uu, u_true, F

def poisson2d_fmultigauss_bcs_high_order(mesh, c_list, s_list, num_gaussians=1, rand_gaussians=False, bc_type="u_true"):
    '''solves Poisson's equation on a given mesh with f MULTI gaussian and b=0
    rand_gaussians: if True, sample num_gaussians centers and scales from a uniform distribution
    :param bc_type: '''

    V = FunctionSpace(mesh, "CG", 3) # Piecewise linear splines

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx # weak form of Laplacian

    x = SpatialCoordinate(mesh)
    # u_true = Function(V, name="u_true") # True solution (Gaussian with extra factor to ensure zero Dirichlet BC)
    # F = Function(V, name="f")

    # Initialize the true solution and forcing term
    u_true_total = 0.
    F_total = 0.

    if rand_gaussians:
        num_gauss = num_gaussians
    else:
        num_gauss = len(c_list)

    for i in range(num_gauss):
        c = Constant(c_list[i])  # Ensure c is a Constant
        s = Constant(s_list[i])  # Ensure s is a Constant

        # Compute the Gaussian and its Laplacian
        gaussian = exp(-(x[0]-c[0])**2/s[0]**2-(x[1]-c[1])**2/s[1]**2)
        laplacian = -(1 / (s[0]**4 * s[1]**4)) * exp(-((c[0] - x[0])**2 / s[0]**2) - (c[1] - x[1])**2 / s[1]**2) \
                    * (4 * c[1]**2 * s[0]**4 - 2 * s[0]**2 * s[1]**4 + 4 * s[1]**4 * (c[0] - x[0])**2 - 8 * c[1] * s[0]**4 * x[1] - 2 * s[0]**4 * (s[1]**2 - 2 * x[1]**2))

        # Compute exact gradient
        grad_gaussian_x = -2 * (x[0] - c[0]) / s[0] ** 2 * gaussian
        grad_gaussian_y = -2 * (x[1] - c[1]) / s[1] ** 2 * gaussian
        # Compute exact Hessian
        hessian_gaussian_xx = (4 * (x[0] - c[0]) ** 2 / s[0] ** 4 - 2 / s[0] ** 2) * gaussian
        hessian_gaussian_yy = (4 * (x[1] - c[1]) ** 2 / s[1] ** 4 - 2 / s[1] ** 2) * gaussian
        hessian_gaussian_xy = 2 * (x[0] - c[0]) * (x[1] - c[1]) / (s[0] ** 2 * s[1] ** 2) * gaussian

        # Add the Gaussian and its Laplacian to the total solution and forcing term
        u_true_total += gaussian
        F_total += laplacian

    # Interpolate the total solution and forcing term
    #u_true.interpolate(u_true_total)
    #F.interpolate(F_total)

    # Project the forcing function into the function space
    u_true = project(u_true_total, V, name="u_true")
    F = project(F_total, V, name="f")

    # Define boundary conditions
    if bc_type == "u_true":
        bc = DirichletBC(V, u_true, "on_boundary")
    elif bc_type == "zero":
        bc = DirichletBC(V, 0, "on_boundary")
    bcs = [bc]

    # Initialise the approximate solution and solve with preconditioning
    uu = Function(V, name="uu")
    uu.assign(0)

    solve(a == F * v * dx, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    return uu, u_true, F


def poisson2d_fmultigauss_b0_derivs(mesh, c_list, s_list, num_gaussians=1, rand_gaussians=False):
    '''solves Poisson's equation on a given mesh with f MULTI gaussian and b=0
    rand_gaussians: if True, sample num_gaussians centers and scales from a uniform distribution'''

    V = FunctionSpace(mesh, "CG", 1) # Piecewise linear splines

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx # weak form of Laplacian
    x = SpatialCoordinate(mesh)
    u_true = Function(V, name="u_true") # True solution (Gaussian with extra factor to ensure zero Dirichlet BC)
    F = Function(V, name="f")

    # Initialize the true solution and forcing term
    u_true_total = 0
    F_total = 0

    if rand_gaussians:
        num_gauss = num_gaussians
    else:
        num_gauss = len(c_list)

    u_gradx_exact_tot = 0
    u_grady_exact_tot = 0
    u_hessxx_exact_tot = 0
    u_hessyy_exact_tot = 0
    u_hessxy_exact_tot = 0
    u_gradx_exact = Function(V, name="u_gradx_exact")
    u_grady_exact = Function(V, name="u_grady_exact")
    u_hessxx_exact = Function(V, name="u_hessxx_exact")
    u_hessyy_exact = Function(V, name="u_hessyy_exact")
    u_hessxy_exact = Function(V, name="u_hessxy_exact")

    for i in range(num_gauss):
        c = c_list[i]
        s = s_list[i]

        # Compute the Gaussian and its Laplacian
        gaussian = exp(-(x[0]-c[0])**2/s[0]**2-(x[1]-c[1])**2/s[1]**2)
        laplacian = -(1 / (s[0]**4 * s[1]**4)) * exp(-((c[0] - x[0])**2 / s[0]**2) - (c[1] - x[1])**2 / s[1]**2) \
                    * (4 * c[1]**2 * s[0]**4 - 2 * s[0]**2 * s[1]**4 + 4 * s[1]**4 * (c[0] - x[0])**2 - 8 * c[1] * s[0]**4 * x[1] - 2 * s[0]**4 * (s[1]**2 - 2 * x[1]**2))

        # Compute exact gradient
        grad_gaussian_x = -2 * (x[0] - c[0]) / s[0] ** 2 * gaussian
        grad_gaussian_y = -2 * (x[1] - c[1]) / s[1] ** 2 * gaussian
        # Compute exact Hessian
        hessian_gaussian_xx = (4 * (x[0] - c[0]) ** 2 / s[0] ** 4 - 2 / s[0] ** 2) * gaussian
        hessian_gaussian_yy = (4 * (x[1] - c[1]) ** 2 / s[1] ** 4 - 2 / s[1] ** 2) * gaussian
        hessian_gaussian_xy = 2 * (x[0] - c[0]) * (x[1] - c[1]) / (s[0] ** 2 * s[1] ** 2) * gaussian

        # Add the Gaussian and its Laplacian to the total solution and forcing term
        u_true_total += gaussian
        F_total += laplacian
        u_gradx_exact_tot += grad_gaussian_x
        u_grady_exact_tot += grad_gaussian_y
        u_hessxx_exact_tot += hessian_gaussian_xx
        u_hessyy_exact_tot += hessian_gaussian_yy
        u_hessxy_exact_tot += hessian_gaussian_xy

    pde_params = {'centers': c_list, 'scales': s_list}

    # Interpolate the total solution and forcing term
    #u_true.interpolate(u_true_total)
    #F.interpolate(F_total)
    
    # Project the forcing function into the function space
    u_true=project(u_true_total, V)
    F = project(F_total, V)

    u_gradx_exact.interpolate(u_gradx_exact_tot)
    u_grady_exact.interpolate(u_grady_exact_tot)
    u_hessxx_exact.interpolate(u_hessxx_exact_tot)
    u_hessyy_exact.interpolate(u_hessyy_exact_tot)
    u_hessxy_exact.interpolate(u_hessxy_exact_tot)

    # Define boundary conditions
    bc = DirichletBC(V, u_true, "on_boundary")
    bcs = [bc]

    # Initialise the approximate solution and solve with preconditioning
    uu = Function(V, name="uu")
    uu.assign(0)

    solve(a == F * v * dx, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    #%% Evaluate the L2 error
    print('\n\nFinished solve, L^2 error is:')
    print(sqrt(assemble(inner(uu - u_true, uu - u_true) * dx)))
    #%% Evaluate the H1 error..

    #calc approx grad
    u_gradx_approx = Function(V, name="u_gradx_approx")
    u_grady_approx = Function(V, name="u_grady_approx")
    u_gradx_approx.interpolate(grad(uu)[0])
    u_grady_approx.interpolate(grad(uu)[1])
    #calc approx hess
    u_hessxx_approx = Function(V, name="u_hessxx_approx")
    u_hessyy_approx = Function(V, name="u_hessyy_approx")
    u_hessxy_approx = Function(V, name="u_hessxy_approx")
    u_hessxx_approx.interpolate(grad(grad(uu)[0])[0])
    u_hessyy_approx.interpolate(grad(grad(uu)[1])[1])
    u_hessxy_approx.interpolate(grad(grad(uu)[0])[1])

    derivs_dict = {'u_gradx_approx': u_gradx_approx, 'u_grady_approx': u_grady_approx,
                      'u_hessxx_approx': u_hessxx_approx, 'u_hessyy_approx': u_hessyy_approx,
                        'u_hessxy_approx': u_hessxy_approx,
                   'u_gradx_exact': u_gradx_exact, 'u_grady_exact': u_grady_exact,
                      'u_hessxx_exact': u_hessxx_exact, 'u_hessyy_exact': u_hessyy_exact,
                        'u_hessxy_exact': u_hessxy_exact}

    return uu, u_true, F, pde_params, derivs_dict


def poisson2d_fsin_b0(mesh):
    '''solves Poisson's equation on a given mesh with f = sin(pi*x[0])*sin(pi*x[1]) and b = 0'''
    pde_params = {'f': 'sin(pi*x[0])*sin(pi*x[1])', 'b': 0}

    # Define function space
    V = FunctionSpace(mesh, "CG", 1)

    # Define boundary condition
    bc = DirichletBC(V, Constant(0), "on_boundary")

    # Define variational problem
    uu = Function(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate(sin(pi*x) * sin(pi*y))  # adjust as needed
    F = (dot(grad(uu), grad(v)) - f * v) * dx

    # Solve problem
    solve(F == 0, uu, bcs=bc, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi'})

    # the exact solution is u_true = -(1 / (2 * pi ^ 2)) * sin(pi * x) * sin(pi * y).
    # This is because the Laplacian of u_true gives
    # f: Δu_true = Δ((1 / (2 * pi ^ 2)) * sin(pi * x) * sin(pi * y)) = sin(pi * x) * sin(pi * y) = f.
    u_true = Function(V)
    u_true.interpolate(-(1 / (2 * pi * pi)) * sin(pi * x) * sin(pi * y))
    #nb this assumes nice domain

    return uu, u_true, F, pde_params


def poisson2d_f0_bsin_polarL(mesh):
    '''solves Poisson's equation on an L shape mesh with f = 0 and b = sin(pi*theta/w)'''
    pde_params = {'f': 0, 'b': 'sin(pi*theta/w)'}

    # Set the value of w
    w = 3. * pi / 2. #1.0  # Adjust this value as needed

    # Define function space
    V = FunctionSpace(mesh, "CG", 1)

    # Define boundary condition
    x, y = SpatialCoordinate(mesh)
    r, theta = cart2pol(x, y)
    # u_D = r ** (pi / w) * sin(pi * abs(theta) / w)
    u_D = r ** (pi / w) * sin(pi * theta / w)
    bc = DirichletBC(V, u_D, "on_boundary")

    # Define variational problem
    uu = Function(V)
    v = TestFunction(V)
    f = Constant(0)  # Zero forcing function
    F = (dot(grad(uu), grad(v)) - f * v) * dx

    # Solve problem
    solve(F == 0, uu, bcs=bc, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi'})

    u_true = None

    # return mesh, u
    return uu, u_true, F, pde_params


# Convert the Cartesian coordinates to polar coordinates
def cart2pol(x, y):
    r = sqrt(x**2 + y**2)
    theta = atan_2(y, x)
    # theta = (theta + 2*pi) % (2*pi)  # Adjust the range of theta from [0, 2*pi]
    theta = conditional(theta < 0, theta + 2*pi, theta)  # Adjust the range of theta from [0, 2*pi]
    return r, theta


def plot_solutions(uu, u_true):

    #%% Plot the solution
    fig, axes = plt.subplots()
    colors = tripcolor(uu, axes=axes)
    plt.title('Approximate solution')
    fig.colorbar(colors)
    #make tight_layout
    plt.tight_layout()


    fig, axes = plt.subplots()
    colors = tripcolor(u_true, axes=axes)
    plt.title('Exact solution')
    fig.colorbar(colors)
    plt.tight_layout()

    # fig, axes = plt.subplots()
    # colors = tricontour(u_true, axes=axes)
    # plt.title('Exact solution contour')
    # fig.colorbar(colors)

    plt.show()

if __name__ == "__main__":
    mesh = UnitSquareMesh(10, 10)
    uu, u_true, F, pde_params = poisson2d_fmultigauss_bcs(mesh, num_gaussians=2, rand_gaussians=True)
    plot_solutions(uu, u_true)
