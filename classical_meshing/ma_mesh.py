import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from create_gmesh import mesh_to_msh_file, plot_test_poisson_mesh, load_mesh_and_solve_poisson


def mesh1(r, B, C, ss, gamma):
    return r**2 + B*r**(2*(1-gamma)) - C*ss**2

#equation 38
# scale
# skew
# smoothness H&R - ie small element next to big element is bad


def mesh2(N, xi, eta, B, C, gamma, LorSq='L'):
    '''takes a square mesh and returns either square or L shape by reflection about both axis
    - returns deformed coordinates'''
    K = 1
    if LorSq == 'L':
        xx = np.ones(3 * N * N)
        yy = np.ones(3 * N * N)
    else:
        xx = np.ones(N * N)
        yy = np.ones(N * N)

    idxs_to_remove = []
    test1, test2, test3 = [], [], []
    for i in range(N * N):
        ss = np.sqrt(xi[i] ** 2 + eta[i] ** 2)

        # if xi[i] >= eta[i]:
        #     L = K * ss / xi[i]
        # else:
        #     L = K * ss / eta[i]
        #
        # if ss < 0.0001:
        #     L = K

        # Check if ss is less than a small threshold before doing any division
        if ss < 0.0001:
            L = K
        elif xi[i] >= eta[i]:
            L = K * ss / xi[i]
        else:
            L = K * ss / eta[i]

        C = 1 + B * L ** (-2 * gamma)
        r = fsolve(mesh1, 0.5, args=(B, C, ss, gamma))

        if ss < 0.0001:
            xx[i] = 0;
            yy[i] = 0;
        else:
            xx[i] = abs(r) * xi[i] / ss;
            yy[i] = abs(r) * eta[i] / ss;

        if LorSq == 'L':
            #reflect in the y axis
            xx[N ** 2 + i] = -xx[i];
            yy[N ** 2 + i] = yy[i];
            #remove everything where x==0 expect N
            if np.isclose(xi[i], 0):
                test1.append((i,N ** 2 + i))
                idxs_to_remove.append(N ** 2 + i)
            #remove everything where y==0
            if np.isclose(eta[i], 0):
                test2.append((i,N ** 2 + i))
                idxs_to_remove.append(N ** 2 + i)

            #reflect in the y and x axis == rotate 180 degrees
            xx[2 * N ** 2 + i] = -xx[i];
            yy[2 * N ** 2 + i] = -yy[i];

            #remove the origin have rotated about
            if np.isclose(eta[i], 0) and np.isclose(xi[i], 0):
                test3.append((i,2 * N ** 2 + i))
                idxs_to_remove.append(2 * N ** 2 + i)

    if LorSq == 'L':
        xx = np.delete(xx, idxs_to_remove)
        yy = np.delete(yy, idxs_to_remove)

    return xx, yy


def meshgen(N, K=1):
    h = K / (N - 1)

    # create nodes
    xi = np.zeros(N * N)
    eta = np.zeros(N * N)

    for j in range(N):
        for i in range(N):
            ix = i + N * j
            xi[ix] = i * h
            eta[ix] = j * h

    # create edges
    horiz = np.zeros((N*N, 4), dtype=int)
    vert = np.zeros((N*N, 4), dtype=int)
    corn1 = np.zeros((N*N, 4), dtype=int)
    corn2 = np.zeros((N*N, 4), dtype=int)

    for j in range(N):
        for i in range(N):
            n = i + N * j

            if i < N - 1:
                # Horizontal edges
                horiz[n, :] = [i, i+1, j, j]
            else:
                horiz[n, :] = [i, i, j, j]

            if j < N - 1:
                # Vertical edges
                vert[n, :] = [i, i, j, j+1]
            else:
                vert[n, :] = [i, i, j, j]

            if j == 0:
                corn1[n, :] = [i, i, j, j]
            else:
                corn1[n, :] = [i, i+1, j, j-1]

            if j == N - 1:
                corn2[n, :] = [i, i, j, j]
            else:
                corn2[n, :] = [i, i+1, j, j+1]

            if i == N - 1:
                corn1[n, :] = [i, i, j, j]
                corn2[n, :] = [i, i, j, j]

    # create elements
    elements = []

    for j in range(N - 1):
        for i in range(N - 1):
            # indices of the nodes in the current square
            n0 = i + N * j
            n1 = n0 + 1
            n2 = n0 + N
            n3 = n2 + 1

            # # create four elements for the current square
            # elements.append([n0, n1, n2])  # triangle 0,1,2
            # elements.append([n0, n2, n3])  # triangle 0,2,3
            # elements.append([n1, n2, n3])  # triangle 1,2,3
            # elements.append([n0, n1, n3])  # triangle 0,1,3

            # create two elements (triangles) for the current square
            # elements.append([n0, n1, n2])  # triangle 0,1,2
            # elements.append([n1, n2, n3])  # triangle 1,2,3

            # create single element (quadrilateral) for the current square
            # elements.append([n0, n1, n3, n2])
            elements.append([n0 + 1, n1 + 1, n3 + 1, n2 + 1]) # Gmsh uses 1-based indexing

    # print max min values of elements
    print(np.max(elements))
    print(np.min(elements))

    return xi, eta, vert, horiz, corn1, corn2, elements

def meshplot(N, xx, yy, vert, corn1, corn2, horiz, nodes=None, boundary_nodes=None):
    plt.figure()

    for i in range(N * N):
        # For vertical edges
        cc = vert[i]
        m1 = cc[0] + N * cc[2]
        m2 = cc[1] + N * cc[3]
        x = xx[m1] + (xx[m2] - xx[m1]) * np.linspace(0, 1, 100)
        y = yy[m1] + (yy[m2] - yy[m1]) * np.linspace(0, 1, 100)
        plt.plot(x, y, 'k')
        plt.plot(-x, y, 'k')
        plt.plot(-x, -y, 'k')

        # For corn2 edges
        cc = corn2[i]
        m1 = cc[0] + N * cc[2]
        m2 = cc[1] + N * cc[3]
        x = xx[m1] + (xx[m2] - xx[m1]) * np.linspace(0, 1, 100)
        y = yy[m1] + (yy[m2] - yy[m1]) * np.linspace(0, 1, 100)
        plt.plot(x, y, 'k')
        plt.plot(-x, y, 'k')
        plt.plot(-x, -y, 'k')

        # For corn1 edges
        cc = corn1[i]
        m1 = cc[0] + N * cc[2]
        m2 = cc[1] + N * cc[3]
        x = xx[m1] + (xx[m2] - xx[m1]) * np.linspace(0, 1, 100)
        y = yy[m1] + (yy[m2] - yy[m1]) * np.linspace(0, 1, 100)
        plt.plot(x, y, 'k')
        plt.plot(-x, y, 'k')
        plt.plot(-x, -y, 'k')

        # For horizontal edges
        cc = horiz[i]
        m1 = cc[0] + N * cc[2]
        m2 = cc[1] + N * cc[3]
        x = xx[m1] + (xx[m2] - xx[m1]) * np.linspace(0, 1, 100)
        y = yy[m1] + (yy[m2] - yy[m1]) * np.linspace(0, 1, 100)
        plt.plot(x, y, 'k')
        plt.plot(-x, y, 'k')
        plt.plot(-x, -y, 'k')

    plt.axis('equal')
    if nodes is not None:
        # Plot boundary nodes
        # boundary_coords = nodes[boundary_nodes]
        boundary_coords = nodes[[n - 1 for n in boundary_nodes]]
        plt.scatter(boundary_coords[:, 0], boundary_coords[:, 1], color='r')

    plt.show()


def edges_from_nodes(node_list):
    # Generate list of edges from a list of nodes
    return [(node_list[i], node_list[i + 1]) for i in range(len(node_list) - 1)]

def get_boundary_nodes(xx, yy, LorSq='L'):

    # Nodes on the outer edges
    x_negative_1_nodes = [i + 1 for i in range(len(xx)) if np.isclose(xx[i], -1)]
    y_positive_1_nodes = [i + 1 for i in range(len(yy)) if np.isclose(yy[i], 1)]
    x_positive_1_nodes = [i + 1 for i in range(len(xx)) if np.isclose(xx[i], 1)]
    y_negative_1_nodes = [i + 1 for i in range(len(yy)) if np.isclose(yy[i], -1)]

    if LorSq == 'L':
        # Nodes on the internal edges of the L-shape
        y_zero_x_positive_nodes = [i + 1 for i in range(len(xx)) if np.isclose(yy[i], 0) and xx[i] >= 0]
        x_zero_y_negative_nodes = [i + 1 for i in range(len(yy)) if np.isclose(xx[i], 0) and yy[i] <= 0]

        outer_edges_nodes = x_negative_1_nodes + y_positive_1_nodes + x_positive_1_nodes + y_negative_1_nodes
        internal_edges_nodes = y_zero_x_positive_nodes + x_zero_y_negative_nodes

    elif LorSq == 'Sq':
        # Nodes for x=0 - testing with one square only
        x_zero_nodes = [i + 1 for i in range(len(xx)) if np.isclose(xx[i], 0)]
        y_zero_nodes = [i + 1 for i in range(len(yy)) if np.isclose(yy[i], 0)]
        outer_edges_nodes = y_positive_1_nodes + x_positive_1_nodes
        internal_edges_nodes = x_zero_nodes + y_zero_nodes

    boundary_nodes = outer_edges_nodes + internal_edges_nodes
    boundary_nodes = list(set(boundary_nodes))

    # Define the boundary edges for each boundary separately
    x_negative_1_edges = edges_from_nodes(x_negative_1_nodes)
    y_positive_1_edges = edges_from_nodes(y_positive_1_nodes)
    x_positive_1_edges = edges_from_nodes(x_positive_1_nodes)
    y_negative_1_edges = edges_from_nodes(y_negative_1_nodes)

    if LorSq == 'L':
        y_zero_x_positive_edges = edges_from_nodes(y_zero_x_positive_nodes)
        x_zero_y_negative_edges = edges_from_nodes(x_zero_y_negative_nodes)
        boundary_edges = x_negative_1_edges + y_positive_1_edges + x_positive_1_edges + y_negative_1_edges \
                         + y_zero_x_positive_edges + x_zero_y_negative_edges
    elif LorSq == 'Sq':
        x_zero_edges = edges_from_nodes(x_zero_nodes)
        y_zero_edges = edges_from_nodes(y_zero_nodes)
        boundary_edges = x_negative_1_edges + y_positive_1_edges + x_positive_1_edges + y_negative_1_edges \
                         + x_zero_edges + y_zero_edges

    return boundary_nodes, boundary_edges


def main(B, C, gamma, N):
    # options = {'disp': False}

    #generate mesh for top right square
    xi, eta, vert, horiz, corn1, corn2, elements = meshgen(N)

    #generate mesh for L shape or the top right square
    LorSq = 'L'#'Sq'# 'L'
    xx, yy = mesh2(N, xi, eta, B, C, gamma, LorSq=LorSq)

    meshplot(N, xx, yy, vert, corn1, corn2, horiz)

    fname = "TestLmesh.msh"
    nodes = np.vstack((xx, yy)).T
    boundary_nodes, boundary_edges = get_boundary_nodes(xx, yy, LorSq=LorSq)

    meshplot(N, xx, yy, vert, corn1, corn2, horiz, nodes, boundary_nodes)

    if LorSq == 'L':
        #todo elements is only built use for the square not the L shape  v2
        raise NotImplementedError
    elif LorSq == 'Sq':
        quad_or_tri = 'quad'
        mesh_to_msh_file(nodes, elements, quad_or_tri, boundary_nodes, fname, boundary_edges)

    # mesh, u = poisson_mesh_test(fname)
    eqn_type = 'zb'
    mesh, uu, u_true, F, pde_params = load_mesh_and_solve_poisson(fname, eqn_type)
    plot_test_poisson_mesh(mesh, uu)


if __name__ == "__main__":
    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Basic Parameters
    % ================
    %
    % N:  Number of points in each direction along the L
    % B:  Compression of the mesh close to the corner.
    % C:  Global mesh scale factor
    % gamma: Mesh scaling factor (this depends on which norm you optimise in)
    % 
    % (xi,eta):  Mesh points in the computational domain
    % (xx,yy):   Mesh points in the physical domain
    % corner1, corner2, horiz, vert: Edge setsin the mesh
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %  Representative values
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    global N
    
    % B = 1   gives a scaled mesh  
    % B = 0   gives a uniform mesh
    '''
    B = 1
    C = 1
    gamma = 2 / 3
    N = 14

    #todo warning node indexing starts at 1 because of matlab
    main(B, C, gamma, N)