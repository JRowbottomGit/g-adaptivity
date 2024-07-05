import time
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from firedrake import TestFunction, TrialFunction, FunctionSpace, Function, DirichletBC, SpatialCoordinate, UnitSquareMesh
from firedrake import inner, grad, dx, div, exp, triplot, tripcolor, solve, sqrt, assemble, tricontour
from torchquad import Simpson,Trapezoid, Gaussian, set_up_backend, utils
from firedrake_difFEM.solve_poisson import poisson2d_fmultigauss_bcs#, u_true_exact_2d
from scipy import sparse


#%% Define FEM basis functions (linear & piecewise cts splines)
# Some auxiliary functions to check if (x,y) is to the left, right or on the line passing through a,b
def checkleft(x,a,b):
    return ((a[1]-b[1])*x[0]+(b[0]-a[0])*x[1]>=(a[1]-b[1])*a[0]+(b[0]-a[0])*a[1])*1.0

def checkright(x,a,b):
    return ((a[1]-b[1])*x[0]+(b[0]-a[0])*x[1]<=(a[1]-b[1])*a[0]+(b[0]-a[0])*a[1])*1.0

def checkon(x,a,b):
    return ((a[1]-b[1])*x[0]+(b[0]-a[0])*x[1]==(a[1]-b[1])*a[0]+(b[0]-a[0])*a[1])*1.0

def aux(x, a, b, c): # Auxiliary function defining value in a single element
    return (checkleft(x,a,b)*checkleft(x,b,c)*checkleft(x,c,a)+checkright(x,a,b)*checkright(x,b,c)*checkright(x,c,a))*(1+((x[0]-c[0])*(a[1]-b[1])+(x[1]-c[1])*(b[0]-a[0]))/((a[1]-b[1])*(c[0]-a[0])+(c[1]-a[1])*(b[0]-a[0])))

def phim(x, n, coords, cell_node_map): # Basis functions (need to correct values along edges hence the various cases below)
    # cell_locations = np.where(cell_node_map == n)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'mps'
    cell_locations = torch.where(cell_node_map == n)
    cell_locations[0].to(device)
    cell_locations[1].to(device)

    #print(cell_node_map)
    #print(n)
    #input('test')
    output=x[0]*0.0
    repeat=x[0]*0.0 # Variable that counts repeat summation of contribution at point x
    # for i in range(np.shape(cell_locations)[1]):
    for i in range(cell_locations[0].shape[0]):
        cell_index=cell_locations[0][i]
        node_index=cell_locations[1][i]

        c_index=cell_node_map[cell_index][node_index]
        # a_index=cell_node_map[cell_index][np.mod(node_index-1,3)]
        # b_index=cell_node_map[cell_index][np.mod(node_index-2,3)]
        a_index=cell_node_map[cell_index][torch.fmod(node_index-1,3).to(device)]
        b_index=cell_node_map[cell_index][torch.fmod(node_index-2,3).to(device)]

        c=coords[c_index]
        a=coords[a_index]
        b=coords[b_index]

        increment=aux(x, a, b, c)

        output = output + increment
        repeat = repeat + (increment > 0.0)*1.0 # Check how often we add a contribution to fix edge and corner values

    return output/(repeat+1.0*(repeat==0.0)) # Correct any repeated contributions on edges and corners


def build_mass_matrix(mesh, mesh_points, num_meshpoints, opt):
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes # Get the boundary nodes
    # Assuming you have a Firedrake mesh object named 'mesh'
    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity
    # Use the cell_node_map to index into the coordinates directly in pytorch
    triangles_tensor = torch.stack([mesh_points[cell] for cell in cell_node_map]).to(opt['device'])#torch.tensor(triangles, dtype=torch.float32)
    # Number of triangles
    T = triangles_tensor.shape[0]
    # Create the matrix A for each triangle
    # This involves creating a [T, 3, 3] tensor where each 3x3 matrix is formed by
    # appending a column of ones to the coordinates of the triangle's vertices
    ones = torch.ones(T, 3, 1).to(opt['device'])
    A = torch.cat((ones, triangles_tensor), dim=2)  # Shape: [T, 3, 3]
    # Create the target matrix B
    # For each triangle's basis function, it should be 1 at its vertex and 0 at the others
    B = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).to(opt['device'])
    B = B.repeat(T, 1, 1)  # Shape: [T, 3, 3]
    # Solve the linear system AX = B for each triangle
    # PyTorch's batched linear solver will handle these systems in parallel
    slopes = torch.linalg.solve(A, B)  # Shape: [T, 3, 3]
    # triangles_tensor is of shape [T, 3, 2] containing the coordinates
    # Extract x and y coordinates
    x = triangles_tensor[:, :, 0]
    y = triangles_tensor[:, :, 1]
    # Calculate the area using the determinant method
    area = 0.5 * torch.abs(x[:,0] * (y[:,1] - y[:,2]) + x[:,1] * (y[:,2] - y[:,0]) + x[:,2] * (y[:,0] - y[:,1]))

    # slopes is of shape [T, 3, 2]
    # Index pairs for dot product
    i_idx = torch.from_numpy(cell_node_map[:, 0])
    j_idx = torch.from_numpy(cell_node_map[:, 1])
    k_idx = torch.from_numpy(cell_node_map[:, 2])

    s_i = slopes[:, 1:, 0]
    s_j = slopes[:, 1:, 1]
    s_k = slopes[:, 1:, 2]

    # Create the sparse Mass matrix
    Mii = (s_i * s_i * area.unsqueeze(1)).sum(1)
    Mjj = (s_j * s_j * area.unsqueeze(1)).sum(1)
    Mkk = (s_k * s_k * area.unsqueeze(1)).sum(1)
    Mij = (s_i * s_j * area.unsqueeze(1)).sum(1)
    Mjk = (s_j * s_k * area.unsqueeze(1)).sum(1)
    Mki = (s_k * s_i * area.unsqueeze(1)).sum(1)

    row_idx = torch.cat((i_idx, j_idx, k_idx, i_idx, j_idx, k_idx, j_idx, k_idx, i_idx)).to(opt['device'])
    col_idx = torch.cat((i_idx, j_idx, k_idx, j_idx, k_idx, i_idx, i_idx, j_idx, k_idx)).to(opt['device'])
    values = torch.cat((Mii, Mjj, Mkk, Mij, Mjk, Mki, Mij, Mjk, Mki))

    M = -torch.sparse_coo_tensor(torch.stack((row_idx, col_idx)), values, size=((num_meshpoints) ** 2, (num_meshpoints) ** 2)).to(opt['device'])
    # Minus comes from integration by parts

    return M, triangles_tensor, slopes


def identify_triangle(triangles_tensor, quadrature_points):
    # triangles_tensor is of shape [T, 3, 2], containing the coordinates of the triangles
    # quadrature_points is of shape [2, Q], containing the coordinates of the quadrature points

    # Extract the vertices of the triangles
    x_i, y_i = triangles_tensor[:, 0, 0], triangles_tensor[:, 0, 1]
    x_j, y_j = triangles_tensor[:, 1, 0], triangles_tensor[:, 1, 1]
    x_k, y_k = triangles_tensor[:, 2, 0], triangles_tensor[:, 2, 1]

    # Compute edge vectors
    e_ij = torch.stack([x_j - x_i, y_j - y_i], dim=-1)
    e_jk = torch.stack([x_k - x_j, y_k - y_j], dim=-1)
    e_ki = torch.stack([x_i - x_k, y_i - y_k], dim=-1)

    # Broadcast and subtract to get point-to-vertex vectors for all quadrature points
    #quadrature_points.T.unsqueeze(1), triangles_tensor[:, 0, :].unsqueeze(0), quadrature_points.T.unsqueeze(1) - triangles_tensor[:, 0, :].unsqueeze(0)
    # p_i = quadrature_points - triangles_tensor[:, 0, :]
    # p_j = quadrature_points - triangles_tensor[:, 1, :]
    # p_k = quadrature_points - triangles_tensor[:, 2, :]
    p_i = quadrature_points.unsqueeze(1) - triangles_tensor[:, 0, :].unsqueeze(0)
    p_j = quadrature_points.unsqueeze(1) - triangles_tensor[:, 1, :].unsqueeze(0)
    p_k = quadrature_points.unsqueeze(1) - triangles_tensor[:, 2, :].unsqueeze(0)

    # Compute cross products (in 2D, this is just the determinant)
    c_ij = e_ij[:, 0] * p_i[:, :, 1] - e_ij[:, 1] * p_i[:, :, 0]
    c_jk = e_jk[:, 0] * p_j[:, :, 1] - e_jk[:, 1] * p_j[:, :, 0]
    c_ki = e_ki[:, 0] * p_k[:, :, 1] - e_ki[:, 1] * p_k[:, :, 0]

    # Check if all cross products have the same sign for each triangle
    inside = (c_ij >= 0) & (c_jk >= 0) & (c_ki >= 0) | (c_ij <= 0) & (c_jk <= 0) & (c_ki <= 0)

    # Find indices of triangles that contain each quadrature point
    # This will give you a boolean tensor of shape [Q, T]
    # where each row corresponds to a quadrature point and each column to a triangle
    triangle_indices = torch.where(inside)

    return triangle_indices


def build_load_vector(mesh, mesh_points, num_meshpoints, n_quad_points_rhs, c_list, s_list, opt):
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes
    coords = mesh_points
    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity
    cell_node_map = torch.tensor(cell_node_map, dtype=torch.long).to(opt['device'])

    RHS = torch.zeros((num_meshpoints) ** 2, 1).to(opt['device'])  # Initialise discrete forcing in FEM
    # Assemble stiffness matrix and automatically incorporate BCs in strong form
    for m in range((num_meshpoints) ** 2):
        # print(f"idx {m}, in BC { m in bc.nodes}")
        if m in bc.nodes:
            RHS[m] = BCfn(torch.tensor([coords[m, 0], coords[m, 1]]).to(opt['device']), c_list, s_list)  # Impose BCs
        else:
            def integrand(x):
                x = torch.transpose(x, 0, 1)
                #print all tensor devices
                # try:
                #     print(f"x {x.device}")
                # except:
                #     pass
                # try:
                #     print(f"m {m.device}")
                # except:
                #     pass
                # try:
                #     print(f"coords {coords.device}")
                # except:
                #     pass
                # try:
                #     print(f"cell_node_map {cell_node_map.device}")
                # except:
                #     pass
                # try:
                #     print(f"{[c.device for c in c_list]}, {[s.device for s in s_list]}")
                # except:
                #     pass

                return phim(x, m, coords, cell_node_map) * f(x, c_list, s_list)

            # RHS[m] = RHS[m] + cubature2d_v2(integrand, n_quad_points_rhs, bounds_support(m, m))
            RHS[m] = RHS[m] + cubature2d_v2(integrand, n_quad_points_rhs, bounds_support_jr(m, coords, cell_node_map))

    return RHS

#todo idea, already calculated the slopes, so can just use those to define phi in the load vector
#loop over the triangles and nodes
def build_load_vector_slopes(mesh, triangles_tensor, slopes, num_meshpoints, n_quad_points_rhs, c, s):
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes
    coords = torch.tensor(mesh.coordinates.dat.data_ro, requires_grad=True).to(opt['device'])
    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity

    #old method
    RHS = torch.zeros((num_meshpoints + 2) ** 2, 1).to(opt['device'])  # Initialise discrete forcing in FEM
    # Assemble stiffness matrix and automatically incorporate BCs in strong form
    for m in range((num_meshpoints + 2) ** 2):
        if m in bc.nodes:
            RHS[m] = BCfn([coords[m, 0], coords[m, 1]], c, s)  # Impose BCs
        else:
            def integrand(x):
                x = torch.transpose(x, 0, 1)
                return phim(x, m, coords, cell_node_map) * f(x, c, s)

            # RHS[m] = RHS[m] + cubature2d_v2(integrand, n_quad_points_rhs, bounds_support(m, m))
            RHS[m] = RHS[m] + cubature2d_v2(integrand, n_quad_points_rhs, bounds_support_jr(m, coords, cell_node_map))

    #new method
    # todo put this inside a function to make just a function of the quadrature points
    def phi_quad(x):#, triangles_tensor, slopes):
        triangle_indices = identify_triangle(triangles_tensor, x)

        coeffs_i = slopes[triangle_indices[1], 0, 0]
        coeffs_j = slopes[triangle_indices[1], 1, 1]
        coeffs_k = slopes[triangle_indices[1], 2, 2]

        # create [1xy] matrix for each quadrature point and multiply with coeffs_i, coeffs_j, coeffs_k
        onexy = torch.stack([torch.ones_like(x[:, 0]), x[:, 0], x[:, 1]], dim=1)
        xyijk = torch.einsum('ijk,ik->ij', onexy, torch.stack([coeffs_i, coeffs_j, coeffs_k], dim=1))

        # phi = phi_i + phi_j + phi_k
        phi = xyijk.sum(1)

        # #eval_phi for each element
        # # phi = coeffs_i * quadrature_points[:, 0] + coeffs_j * quadrature_points[:, 1] + coeffs_k * (1 - quadrature_points[:, 0] - quadrature_points[:, 1])
        # phi_i = coeffs_i * quadrature_points[:, 0]
        # phi_j = coeffs_j * quadrature_points[:, 1]
        # phi_k = coeffs_k * (1 - quadrature_points[:, 0] - quadrature_points[:, 1])

        return phi

    def integrand(x):
        # x = torch.transpose(x, 0, 1)
        return phi_quad(x) * f(torch.transpose(x, 0, 1), c, s)

    RHS = cubature2d_v2(integrand, n_quad_points_rhs, bounds_support_jr(m, coords, cell_node_map))

    return RHS

def f(x, c_list, s_list): # Forcing in Laplace's equation
    # return -2*2*torch.pi*2*torch.pi*torch.sin(2*torch.pi*x)*torch.sin(2*torch.pi*y)
    # return x*0.0+1.0
    sol = torch.zeros(x[0].shape)
    for c, s in zip(c_list, s_list):
        sol += (1 / (s[0]**4 * s[1]**4)) * torch.exp(-((c[0] - x[0])**2 / s[0]**2) - (c[1] - x[1])**2 / s[1]**2) * (4 * c[1]**2 * s[0]**4 - 2 * s[0]**2 * s[1]**4 + 4 * s[1]**4 * (c[0] - x[0])**2 - 8 * c[1] * s[0]**4 * x[1] - 2 * s[0]**4 * (s[1]**2 - 2 * x[1]**2))
    return sol

def u_true_exact_2d(x, c_list, s_list): # True solution (only used for verification later)
    # return torch.sin(2*torch.pi*x)*torch.sin(2*torch.pi*y)
    # return x*(x-1.0)/2.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'mps'

    if isinstance(x, torch.Tensor):
        sol = torch.zeros(x[0].shape).to(device)
        for c, s in zip(c_list, s_list):
            sol += torch.exp(-(x[0]-c[0])**2/s[0]**2-(x[1]-c[1])**2/s[1]**2)
        return sol
    elif isinstance(x, np.ndarray):
        sol = np.zeros(x[0].shape)
        for c, s in zip(c_list, s_list):
            sol += np.exp(-(x[0]-c[0])**2/s[0]**2-(x[1]-c[1])**2/s[1]**2)
        return sol


def BCfn(x, c_list, s_list): # Boundary condition
    return u_true_exact_2d(x, c_list, s_list)#0.0*x+0.0


# Function to compute joint support (or at least a bounding box)
def bounds_support(m,n,Adj,coords):
    nbs=Adj[:,[n]]*Adj[:,[m]]
    indices=nbs.nonzero()[0].to('cpu')
    minvals=np.min(coords[indices].to('cpu').detach().numpy(),0)
    maxvals=np.max(coords[indices].to('cpu').detach().numpy(),0)
    return [[minvals[0],maxvals[0]],[minvals[1],maxvals[1]]]


def bounds_support_jr(m, coords, cell_node_map):
    #indices = cell_node_map[m, :] # Careful not to misuse cell indices for nodes
    # indices=cell_node_map[np.where(cell_node_map == m)[0], :].flatten()
    indices=cell_node_map[torch.where(cell_node_map == m)[0], :].flatten().to('cpu')

    # oldminvals = np.min(coords[indices].to('cpu').detach().numpy(), 0)
    # oldmaxvals = np.max(coords[indices].to('cpu').detach().numpy(), 0)
    minvals = torch.min(coords[indices].to('cpu').detach(), 0)[0]
    maxvals = torch.max(coords[indices].to('cpu').detach(), 0)[0]
    # minvals = torch.min(coords[indices], 0)[0]
    # maxvals = torch.max(coords[indices], 0)[0]
    return [[minvals[0], maxvals[0]], [minvals[1], maxvals[1]]]


def soln(x, out, coords, cell_node_map): # Compute solution values of our approximation
    #x=np.transpose(x1)
    interior=0.0*x[0]
    # for m in range((num_meshpoints)**2):
    for m in range(coords.shape[0]):
        interior=interior+out[m]*phim(x, m, coords, cell_node_map)
    return interior

# #####TODO 1 use element coefficients method to interpolate the solution onto the new mesh
# #####TODO 2 use grid_sample to interpolate the solution onto the new mesh
# def soln_grid_sample(x, out, coords): # Compute solution values of our approximation
#     sol = torch.nn.functional.grid_sample(out.reshape(1, 1, num_meshpoints + 2, num_meshpoints + 2),
#                                           torch.tensor([[[x[0], x[1]]]], dtype=torch.float32),
#                                           mode='bilinear', align_corners=True)
#     sol = interpolate_firedrake_tensor(u, x, mapping_tensor, batch_size, mesh_dims)
#     return sol


def true_interpolated(x, coords, cell_node_map, c, s): # For comparison include the true solution on the interpolated mesh
    interior=0.0*x[0]
    for m in range((num_meshpoints)**2):
        interior=interior+u_true_exact_2d([coords[m,0],coords[m,1]], c, s)*phim(x ,m ,coords, cell_node_map)
    return interior


def cubature2d_v2(integrand, n, domain=[[0, 1], [0, 1]]):
    # Declare an integrator;
    # gauss = Gaussian()
    # traz = Trapezoid()
    simp = Simpson()
    return simp.integrate(integrand,dim=2,N=n,integration_domain=domain,backend="torch",)


def torch_FEM_2D(opt, mesh, mesh_points, quad_points, num_meshpoints, c_list, s_list):
    load_quad_points = opt['load_quad_points']
    # stiff_quad_points = opt['stiff_quad_points'] #todo is it ok this isn't used?
    # num_sol_points = opt['eval_quad_points'] #todo is it ok this isn't used?

    # %% Assembly of linear system
    A, triangles_tensor, slopes = build_mass_matrix(mesh, mesh_points, num_meshpoints, opt)
    A = A.to_dense()

    # Correct to impose Dirichlet BCs in strong sense
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes
    A[bc.nodes, :] = torch.zeros([bc.nodes.size, (num_meshpoints) ** 2]).to(opt['device'])  # Correction term for A to impose BC in strong sense
    A[bc.nodes, bc.nodes] = 1

    #### Assembly of RHS
    RHS = build_load_vector(mesh, mesh_points, num_meshpoints, load_quad_points, c_list, s_list, opt)
    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity
    cell_node_map = torch.tensor(cell_node_map, dtype=torch.long).to(opt['device'])

    # %% Solution of linear system
    coeffs = torch.linalg.solve(A, RHS)
    # solution for coefficients, *without differentiating the matrix solve*

    sol = soln(quad_points, coeffs, mesh_points, cell_node_map)

    return coeffs, mesh_points, sol

def gradient_meshpoints_2D(opt,data, x_phys):
    if 'grad_type' not in opt.keys():
        error_message = "Error: opt['grad_type'] not specified"
        raise ValueError(error_message)
    else:
        if opt['grad_type']=='PDE_loss_direct_mse':
            return gradient_meshpoints_2D_PDE_loss_direct_mse(opt,data, x_phys)
        elif opt['grad_type']=='PDE_loss_direct_L2':
            return gradient_meshpoints_2D_PDE_loss_direct_L2(opt,data, x_phys)
        elif opt['grad_type']=='PDE_loss_adjoint_L2':
            return gradient_meshpoints_2D_PDE_loss_adjoint_L2(opt,data, x_phys)
        else:
            error_message = "Error: opt['grad_type'] incorrectly specified"
        raise ValueError(error_message)

def gradient_meshpoints_2D_PDE_loss_direct_mse(opt, data, x_phys):
    num_meshpoints= opt['mesh_dims'][0]
    mesh = UnitSquareMesh(num_meshpoints - 1, num_meshpoints - 1)
    c_list=data.pde_params['centers'][0]
    s_list=data.pde_params['scales'][0]
    mesh.coordinates.dat.data[:] = x_phys.to('cpu').detach().numpy()
    n_quad_points_rhs=opt['eval_quad_points']
    mesh_points = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=True)
    # %% Assembly of linear system
    A, triangles_tensor, slopes = build_mass_matrix(mesh, mesh_points, num_meshpoints)
    A = A.to_dense()

    # Correct to impose Dirichlet BCs in strong sense
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes
    A[bc.nodes, :] = torch.zeros(
        [bc.nodes.size, (num_meshpoints) ** 2])  # Correction term for A to impose BC in strong sense
    A[bc.nodes, bc.nodes] = 1

    RHS = build_load_vector(mesh, mesh_points, num_meshpoints, n_quad_points_rhs, c_list, s_list, opt)
    # todo this is close but needs more work
    # RHS2 = build_load_vector_slopes(mesh, triangles_tensor, slopes, num_meshpoints, n_quad_points_rhs, c, s)

    # %% Solution of linear system
    out = torch.linalg.solve(A, RHS)
    # coords = torch.tensor(mesh.coordinates.dat.data_ro, requires_grad=True)
    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity

    # partial to make function of x only
    def integrand(x):
        x = torch.transpose(x, 0, 1)
        return (u_true_exact_2d(x, c_list, s_list) - soln(x, out, mesh_points, cell_node_map)) ** 2

    # Introducing articifial recovery of parameters etc
    x0 = torch.linspace(0, 1, opt['eval_quad_points'])  # , dtype=torch.float64)
    y0 = torch.linspace(0, 1, opt['eval_quad_points'])  # , dtype=torch.float64)
    [X, Y] = torch.meshgrid(x0, y0)
    quad_points = [X, Y]
    quad_points=torch.stack(quad_points, dim=0)

    val1=soln(quad_points, out, mesh_points, cell_node_map)


    val2=u_true_exact_2d(quad_points, c_list, s_list)

    loss=F.mse_loss(val1,val2)
    #loss = cubature2d_v2(integrand, n_quad_points_rhs)  # L2 error in approximation

    loss.backward()  # Note this is differentiable wrt meshpoints

    return torch.tensor(loss.to('cpu').detach().numpy()), mesh_points.grad

def gradient_meshpoints_2D_PDE_loss_direct_L2(opt,data, x_phys):
    num_meshpoints= opt['mesh_dims'][0]
    mesh = UnitSquareMesh(num_meshpoints - 1, num_meshpoints - 1)
    c_list=data.pde_params['centers'][0]
    s_list=data.pde_params['scales'][0]
    mesh.coordinates.dat.data[:] = x_phys.to('cpu').detach().numpy()
    n_quad_points_rhs=opt['eval_quad_points']
    mesh_points = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=True)
    # %% Assembly of linear system
    A, triangles_tensor, slopes = build_mass_matrix(mesh, mesh_points, num_meshpoints)
    A = A.to_dense()

    # Correct to impose Dirichlet BCs in strong sense
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes
    A[bc.nodes, :] = torch.zeros(
        [bc.nodes.size, (num_meshpoints) ** 2])  # Correction term for A to impose BC in strong sense
    A[bc.nodes, bc.nodes] = 1

    RHS = build_load_vector(mesh, mesh_points, num_meshpoints, n_quad_points_rhs, c_list, s_list, opt)
    # todo this is close but needs more work
    # RHS2 = build_load_vector_slopes(mesh, triangles_tensor, slopes, num_meshpoints, n_quad_points_rhs, c, s)

    # %% Solution of linear system
    out = torch.linalg.solve(A, RHS)
    # coords = torch.tensor(mesh.coordinates.dat.data_ro, requires_grad=True)
    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity

    # partial to make function of x only
    def integrand(x):
        x = torch.transpose(x, 0, 1)
        return (u_true_exact_2d(x, c_list, s_list) - soln(x, out, mesh_points, cell_node_map)) ** 2

    loss = cubature2d_v2(integrand, n_quad_points_rhs)  # L2 error in approximation

    loss.backward()  # Note this is differentiable wrt meshpoints

    return torch.tensor(loss.to('cpu').detach().numpy()), mesh_points.grad

def gradient_meshpoints_2D_PDE_loss_adjoint_L2(opt, data, x_phys):
    num_meshpoints = opt['mesh_dims'][0]
    mesh = UnitSquareMesh(num_meshpoints - 1, num_meshpoints - 1)
    c_list = data.pde_params['centers'][0]
    s_list = data.pde_params['scales'][0]
    mesh.coordinates.dat.data[:] = x_phys.to('cpu').detach().numpy()
    n_quad_points_rhs = opt['eval_quad_points']
    # Need two copies of mesh_points, since we need to do two backpropagations in adjoint method
    mesh_points = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=True)

    # %% Assembly of linear system
    A, triangles_tensor, slopes = build_mass_matrix(mesh, mesh_points, num_meshpoints)
    A = A.to_dense()

    # Correct to impose Dirichlet BCs in strong sense
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes
    A[bc.nodes, :] = torch.zeros(
        [bc.nodes.size, (num_meshpoints) ** 2])  # Correction term for A to impose BC in strong sense
    A[bc.nodes, bc.nodes] = 1

    #### Assembly of RHS
    RHS = build_load_vector(mesh, mesh_points, num_meshpoints, opt['load_quad_points'], c_list, s_list, opt)
    # todo this is close but needs more work
    # RHS2 = build_load_vector_slopes(mesh, triangles_tensor, slopes, num_meshpoints, n_quad_points_rhs, c, s)

    cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity

    with torch.no_grad():
        out_nograd = torch.tensor(sp.linalg.solve(A.to('cpu').detach().numpy(), RHS.to('cpu').detach().numpy()))
        # solution for coefficients, *without differentiating the matrix solve*

    # outnew = out_nograd.to('cpu').detach().requires_grad_(True)
    outnew = out_nograd.detach().requires_grad_(True)

    # Loss, now differentiable with respect to mesh points only
    def integrand(x):
        x = torch.transpose(x, 0, 1)
        return (u_true_exact_2d(x, c_list, s_list) - soln(x, outnew, mesh_points, cell_node_map)) ** 2

    loss = cubature2d_v2(integrand, opt['load_quad_points'])  # L2 error in approximation
    grad1=torch.autograd.grad(loss, mesh_points, torch.ones_like(loss),retain_graph=True)[0]#loss.backward()  # Note this is differentiable wrt meshpoints
    outnew_grad=torch.autograd.grad(loss, outnew, torch.ones_like(loss),retain_graph=True)[0]

    #lambda1 = torch.tensor(sp.linalg.solve(A.to('cpu').detach().t().numpy(), -outnew.grad.to('cpu').detach().numpy()))
    lambda1 = torch.tensor(sp.linalg.solve(A.to('cpu').detach().t().numpy(), -outnew_grad.to('cpu').detach().numpy()))

    # Finally, compute gradient wrt mesh points
    g = torch.matmul(A, out_nograd).sub_(RHS)  # define g=Aout-RHS
    final = torch.dot(lambda1.to('cpu').detach()[:, 0], g[:, 0])

    grad2=torch.autograd.grad(final, mesh_points,torch.ones_like(final),retain_graph=False)[0]#final.backward()  # Note gradients are automatically summed hence no .grad.zero

    return torch.tensor(loss.to('cpu').detach().numpy()), grad1+grad2

def train_step(num_meshpoints, n_quad_points, n_quad_points_rhs, learning_rate=0.01, epochs=3,c = [0.5, 0.9],s=[0.1, 0.1], opt={}):
    opt['device'] = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'mps'
    start = time.time()
    # Initialise mesh meta data
    mesh = UnitSquareMesh(num_meshpoints - 1, num_meshpoints - 1)

    mesh_points = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=True)

    # Use SGD or Adam optimizer for gradient descent
    # optimizer = torch.optim.Adam([mesh], lr=learning_rate)
    optimizer = torch.optim.SGD([mesh_points], lr=learning_rate)

    for j in range(epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # %% Assembly of linear system
        A, triangles_tensor, slopes = build_mass_matrix(mesh, mesh_points, num_meshpoints)
        A = A.to_dense()

        # Correct to impose Dirichlet BCs in strong sense
        V = FunctionSpace(mesh, "CG", 1)
        bc = DirichletBC(V, 0, "on_boundary")
        bc.nodes  # Get the boundary nodes
        A[bc.nodes, :] = torch.zeros([bc.nodes.size, (num_meshpoints) ** 2])  # Correction term for A to impose BC in strong sense
        A[bc.nodes, bc.nodes] = 1

        RHS = build_load_vector(mesh, mesh_points, num_meshpoints, n_quad_points_rhs, c, s, opt)
        # todo this is close but needs more work
        # RHS2 = build_load_vector_slopes(mesh, triangles_tensor, slopes, num_meshpoints, n_quad_points_rhs, c, s)

        # %% Solution of linear system
        out = torch.linalg.solve(A, RHS)
        # coords = torch.tensor(mesh.coordinates.dat.data_ro, requires_grad=True)
        cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity

        # partial to make function of x only
        def integrand(x):
            x = torch.transpose(x, 0, 1)
            return (u_true_exact_2d(x, c, s) - soln(x, out, mesh_points, cell_node_map)) ** 2

        loss = cubature2d_v2(integrand, n_quad_points_rhs)  # L2 error in approximation

        loss.backward()  # Note this is differentiable wrt meshpoints

        end = time.time()
        print(end - start)

        with torch.no_grad():
            plot_results(out, mesh_points.to('cpu').detach(), cell_node_map, n_quad_points, c, s)

        optimizer.step()

    return mesh_points

def train_step_adjoint(opt, num_meshpoints, c_list, s_list, lr=0.01, epochs=3, plotandsave=False):

    start = time.time()
    # Initialise mesh meta data
    mesh = UnitSquareMesh(num_meshpoints - 1, num_meshpoints - 1)
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.nodes  # Get the boundary nodes

    # Index complement of bc.nodes
    internal_ind =  np.setdiff1d(np.arange((num_meshpoints) ** 2), bc.nodes)

    internal_mesh_points = torch.tensor(mesh.coordinates.dat.data[internal_ind], dtype=torch.float32, requires_grad=True)

    # Need two copies of mesh_points, since we need to do two backpropagations in adjoint method
    mesh_points = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=False)
    mesh_points[internal_ind] = internal_mesh_points
    mesh_points2 = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=False)
    mesh_points2[internal_ind] = internal_mesh_points

    # Use SGD or Adam optimizer for gradient descent
    # optimizer = torch.optim.Adam([mesh], lr=learning_rate)
    optimizer = torch.optim.SGD([internal_mesh_points], lr=lr)

    loss_list, mesh_list = [], []
    for j in range(epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # %% Assembly of linear system
        A, triangles_tensor, slopes = build_mass_matrix(mesh, mesh_points, num_meshpoints, opt)
        A = A.to_dense()

        # Correct to impose Dirichlet BCs in strong sense
        A[bc.nodes, :] = torch.zeros([bc.nodes.size, (num_meshpoints) ** 2])  # Correction term for A to impose BC in strong sense
        A[bc.nodes, bc.nodes] = 1

        #### Assembly of RHS
        RHS = build_load_vector(mesh, mesh_points, num_meshpoints, opt['load_quad_points'], c_list, s_list, opt)
        # todo this is close but needs more work
        # RHS2 = build_load_vector_slopes(mesh, triangles_tensor, slopes, num_meshpoints, n_quad_points_rhs, c, s)

        cell_node_map = mesh.coordinates.cell_node_map().values  # Get cell-node connectivity
        cell_node_map = torch.tensor(cell_node_map, dtype=torch.long).to(opt['device'])

        with torch.no_grad():
            out_nograd = torch.tensor(sp.linalg.solve(A.to('cpu').detach().numpy(), RHS.to('cpu').detach().numpy()))
            # solution for coefficients, *without differentiating the matrix solve*

        # outnew = out_nograd.to('cpu').detach().requires_grad_(True)
        outnew = out_nograd.detach().requires_grad_(True)

        # Loss, now differentiable with respect to mesh points only
        def integrand(x):
            x = torch.transpose(x, 0, 1)
            return (u_true_exact_2d(x, c_list, s_list) - soln(x, outnew, mesh_points2, cell_node_map)) ** 2

        loss = cubature2d_v2(integrand, opt['load_quad_points'])  # L2 error in approximation
        loss.backward()  # Note this is differentiable wrt meshpoints
        print("Iteration:", j, "Loss:", loss.item())

        # Now solve adjoint equation
        # Take transpose of A and solve for A^T lambda = -grad(loss wrt out)
        # mesh_points.grad.zero_() Note gradients are automatically summed hence no .grad.zero
        lambda1 = torch.tensor(sp.linalg.solve(A.to('cpu').detach().t().numpy(), -outnew.grad.to('cpu').detach().numpy()))

        # Finally, compute gradient wrt mesh points
        g = torch.matmul(A, out_nograd).sub_(RHS)  # define g=Aout-RHS
        final = torch.dot(lambda1.to('cpu').detach()[:, 0], g[:, 0])
        final.backward()  # Note gradients are automatically summed hence no .grad.zero
        optimizer.step()
        optimizer.zero_grad()

        # Update mesh points
        mesh_points = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=False)
        mesh_points[internal_ind] = internal_mesh_points
        mesh_points2 = torch.tensor(mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=False)
        mesh_points2[internal_ind] = internal_mesh_points


        end = time.time()

        if plotandsave and j % 5 == 1:
            with torch.no_grad():
                plot_results(outnew.to('cpu').detach(), mesh_points.to('cpu').detach(), cell_node_map, n_quad_points, c_list, s_list)

        loss_list.append(loss.item())
        mesh_list.append(mesh_points.to('cpu').detach().numpy().copy()) #requires .copy as .copy_ and .add_ are in-place operations


    # return mesh_points

    return out_nograd, mesh_points, None, loss_list, mesh_list


class backFEM_2D(torch.nn.Module):
    '''a wrapper for differentiable backFEM solver that minimizes the L2 error of the approximation and returns the updated mesh'''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # self.mesh = torch.nn.Parameter(torch.tensor(np.linspace(0, 1, opt['mesh_dims'][0]), requires_grad=True))

        self.num_meshpoints = opt['mesh_dims'][0] #internal mesh points
        self.lr = opt['lr']
        self.epochs = opt['epochs']

    def forward(self, data):
        c_list = [torch.from_numpy(c) for c in data.pde_params['centers'][0]]
        s_list = [torch.from_numpy(s) for s in data.pde_params['scales'][0]]
        coeffs, MLmodel_coords, sol, loss_list, mesh_list = train_step_adjoint(self.opt, self.num_meshpoints, c_list, s_list, self.lr, self.epochs)

        return coeffs, MLmodel_coords, sol

class Fixed_Mesh_2D(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n = opt['mesh_dims'][0]
        self.m = opt['mesh_dims'][1]
        self.num_meshpoints = self.n * self.m
        self.mesh = UnitSquareMesh(self.n - 1, self.m - 1, name="ref_mesh")
        # self.eval_quad_points = opt['eval_quad_points']
        x_values = np.linspace(0, 1, opt['eval_quad_points'])
        y_values = np.linspace(0, 1, opt['eval_quad_points'])
        # Create the mesh grid
        X, Y = np.meshgrid(x_values, y_values)
        self.eval_grid = torch.from_numpy(np.reshape(np.array([X, Y]), [2, opt['eval_quad_points']**2]))

    def forward(self, data):
        mesh_points = data.x_comp
        if self.opt['loss_type'] == 'mesh_loss':
            return mesh_points

        elif self.opt['loss_type'] == 'pde_loss':
            c_list = [torch.from_numpy(c) for c in data.pde_params['centers'][0]]
            s_list = [torch.from_numpy(s) for s in data.pde_params['scales'][0]]
            coeffs, MLmodel_coords, sol = torch_FEM_2D(self.opt, self.mesh, mesh_points, self.eval_grid, self.n, c_list, s_list)
            return coeffs, MLmodel_coords, sol


def plot_results(out, coords, cell_node_map, n_quad_points, c, s):
    #%% Plot results
    # Coordinates for plotting
    x0=torch.tensor(np.linspace(0,1,n_quad_points),requires_grad=True)
    y0=torch.tensor(np.linspace(0,1,n_quad_points))
    x=[x0,y0]
    [X,Y]=torch.meshgrid(x[0],x[1])

    fig, axes = plt.subplots()
    colors = plt.contour(X.to('cpu').detach().numpy(),Y.to('cpu').detach().numpy(),soln([X,Y], out, coords, cell_node_map).to('cpu').detach().numpy(), axes=axes)
    plt.title('Approximate solution')
    fig.colorbar(colors)

    fig, axes = plt.subplots()
    colors = plt.contour(X.to('cpu').detach().numpy(),Y.to('cpu').detach().numpy(),true_interpolated([X,Y], coords, cell_node_map, c, s).to('cpu').detach().numpy(), axes=axes)
    plt.title('Interpolated exact solution')
    fig.colorbar(colors)

    fig, axes = plt.subplots()
    colors = plt.contour(X.to('cpu').detach().numpy(),Y.to('cpu').detach().numpy(),u_true_exact_2d([X,Y], c, s).to('cpu').detach().numpy(), axes=axes)
    plt.title('Exact solution')
    fig.colorbar(colors)

    plt.show()

    #%% Surface plots for visualisation
    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X.to('cpu').detach().numpy(),Y.to('cpu').detach().numpy(),u_true_exact_2d([X,Y], c, s).to('cpu').detach().numpy(),cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X.to('cpu').detach().numpy(),Y.to('cpu').detach().numpy(),true_interpolated([X,Y], coords, cell_node_map, c, s).to('cpu').detach().numpy(),cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X.to('cpu').detach().numpy(),Y.to('cpu').detach().numpy(),soln([X,Y], out, coords, cell_node_map).to('cpu').detach().numpy(),cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

# if __name__ == '__main__':
#     # Enable GPU support if available and set the floating point precision
#     set_up_backend("torch", data_type="float32")
#     num_meshpoints = 11  # Number of mesh points
#     n_quad_points = 200  # Number of quadrature points (old trapezoidal)
#     # n_quad_points_lhs = 10000  # Number of quadrature points (torchquad terms in stiffness matrix)
#     n_quad_points_rhs = 50000  # Number of quadrature points (torchquad forcing terms)
#
#     train_step(num_meshpoints, n_quad_points, n_quad_points_rhs)

from torchdiffeq import odeint, odeint_adjoint
if __name__ == '__main__':
    # Enable GPU support if available and set the floating point precision
    set_up_backend("torch", data_type="float32")
    num_meshpoints = 11  # Number of mesh points
    n_quad_points = 50  # Number of quadrature points (old trapezoidal)
    # n_quad_points_lhs = 10000  # Number of quadrature points (torchquad terms in stiffness matrix)
    n_quad_points_rhs = 50000  # Number of quadrature points (torchquad forcing terms)

    c_list= [np.array([0.5, 0.9])]
    s_list= [np.array([0.1, 0.1])]
    opt = {'load_quad_points': 50000} #n_quad_points_rhs
    out_nograd, mesh_points, _, loss_list, mesh_list=train_step_adjoint(opt, num_meshpoints, c_list, s_list, 0.1, 20)

    mesh = UnitSquareMesh(num_meshpoints - 1, num_meshpoints - 1)
    mesh.coordinates.dat.data[:] = mesh_points.to('cpu').detach().numpy()

    uu, _ ,_ = poisson2d_fmultigauss_bcs(mesh, c_list, s_list)

    uu(mesh_points.to('cpu').detach().numpy().tolist())

    n_quad_points=10
    x0 = np.linspace(0, 1, n_quad_points)
    y0 = np.linspace(0, 1, n_quad_points)
    x = [x0, y0]
    [X,Y]=np.meshgrid(x[0],x[1])
    x=np.reshape([X,Y],[2,n_quad_points**2])

    approx=np.reshape(uu(np.transpose(x).tolist()),[n_quad_points,n_quad_points])

    fig, axes = plt.subplots()
    colors = plt.contour(X,Y,approx, axes=axes)
    plt.title('Approximate solution firedrake')
    fig.colorbar(colors)
    plt.show()