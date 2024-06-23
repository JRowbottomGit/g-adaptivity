from firedrake import *
from matplotlib import pyplot as plt

from firedrake_difFEM.solve_poisson import poisson2d_fsin_b0, poisson2d_f0_bsin_polarL

def create_Testmesh(fname):
    # Nodes and elements data
    # nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 4 nodes
    # elements = np.array([[1, 2, 3], [1, 3, 4]])  # 2 elements (triangles)
    # boundary_nodes = [1, 2, 3, 4]  # All nodes are on the boundary
    #too simple all nodes on boundary

    nodes = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],  # nodes 1-4
        [0, 1], [1, 1], [2, 1], [3, 1],  # nodes 5-8
        [0, 2], [1, 2], [2, 2], [3, 2],  # nodes 9-12
        [0, 3], [1, 3], [2, 3], [3, 3]  # nodes 13-16
    ])

    elements = np.array([
        [1, 2, 6], [1, 6, 5],  # triangles in square 1
        [2, 3, 7], [2, 7, 6],  # triangles in square 2
        [3, 4, 8], [3, 8, 7],  # triangles in square 3
        [5, 6, 10], [5, 10, 9],  # triangles in square 4
        [6, 7, 11], [6, 11, 10],  # triangles in square 5
        [7, 8, 12], [7, 12, 11],  # triangles in square 6
        [9, 10, 14], [9, 14, 13],  # triangles in square 7
        [10, 11, 15], [10, 15, 14],  # triangles in square 8
        [11, 12, 16], [11, 16, 15]  # triangles in square 9
    ])

    boundary_nodes = [1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5]  # Nodes on the boundary

    # Open file
    with open(fname, "w") as f:

        # # Write mesh format (version 2 ASCII in this case)
        # f.write("$MeshFormat\n2.0 0 8\n$EndMeshFormat\n")

        # Write mesh format (version 2.2 ASCII in this case)
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Write nodes
        f.write("$Nodes\n")
        f.write(str(len(nodes)) + "\n")  # Number of nodes
        for i, node in enumerate(nodes, start=1):  # Gmsh uses 1-based indexing
            f.write(f"{i} {node[0]} {node[1]} 0.0\n")  # Node ID, x, y, z (z=0 for 2D)
        f.write("$EndNodes\n")

        # Write elements
        f.write("$Elements\n")
        # f.write(str(len(elements)) + "\n")  # Number of elements
        f.write(str(len(elements) + len(boundary_nodes)) + "\n")  # Total number of elements (elements + boundary edges)

        for i, element in enumerate(elements, start=1):  # Gmsh uses 1-based indexing
            f.write(f"{i} 2 2 0 1 " + " ".join(map(str, element)) + "\n")  # Elem ID, type (2=triangle), tags (2, 0, 1), node IDs

        # Write boundary elements (edges)
        boundary_tag = 2  # A unique tag to identify the boundary edges in Firedrake
        for i in range(len(boundary_nodes)):
            # Element ID (start from end of last elements and continue), type (1=2-node line)
            # tags (2, boundary_tag, 1), node IDs
            f.write(f"{len(elements) + i + 1} 1 2 {boundary_tag} 1 {boundary_nodes[i]} {boundary_nodes[(i + 1) % len(boundary_nodes)]}\n")


        f.write("$EndElements\n")


def load_mesh_and_solve_poisson(fname, eqn_type='zb'):
    # Load the mesh from file and solve the Poisson equation
    mesh = Mesh(fname)
    if eqn_type == 'zb':
        uu, u_true, F, pde_params = poisson2d_fsin_b0(mesh)
    elif eqn_type == 'sinb':
        uu, u_true, F, pde_params = poisson2d_f0_bsin_polarL(mesh)
    # Save solution to file
    # file = File("../src/solution.pvd")
    # file.write(u)

    return mesh, uu, u_true, F, pde_params

def plot_test_poisson_mesh(mesh, u):
    #%% Plot the solution
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    plt.title('Approximate solution')
    fig.colorbar(colors)
    plt.show()

    #%% Plot the mesh
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    plt.title('Mesh')
    axes.legend()
    plt.show()


def mesh_to_msh_file(nodes, elements, quad_or_tri, boundary_nodes, fname, boundary_edges=None):
    # Open file
    with open(fname, "w") as f:
        # Write mesh format (version 2.2 ASCII in this case)
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Write nodes
        f.write("$Nodes\n")
        f.write(str(len(nodes)) + "\n")  # Number of nodes
        for i, node in enumerate(nodes, start=1):  # Gmsh uses 1-based indexing
            f.write(f"{i} {node[0]} {node[1]} 0.0\n")  # Node ID, x, y, z (z=0 for 2D)
        f.write("$EndNodes\n")

        # Write elements
        f.write("$Elements\n")
        # f.write(str(len(elements)) + "\n")  # Number of elements
        f.write(str(len(elements) + len(boundary_nodes)) + "\n")  # Total number of elements (elements + boundary edges)
        for i, element in enumerate(elements, start=1):  # Gmsh uses 1-based indexing
            # Elem ID, type (3=4-node quadrangle, 2=3-node triangle), tags (2, 0, 1), node IDs
            if quad_or_tri == "quad":
                f.write(f"{i} 3 2 0 1 " + " ".join(map(str, element)) + "\n")
            elif quad_or_tri == "tri":
                f.write(f"{i} 2 2 0 1 " + " ".join(map(str, element)) + "\n")

        # Writing boundary elements (Line in your case)
        boundary_tag = 2  # change this to a unique tag that you'll use to identify the boundary in Firedrake
        if boundary_edges is not None:
            for i, edge in enumerate(boundary_edges, start=1):  # Now iterating over edges
                # Element ID (start from end of last elements and continue), type (1=2-node line)
                # tags (2, boundary_tag, 1), node IDs
                f.write(
                    f"{len(elements) + i} 1 2 {boundary_tag} 1 {edge[0]} {edge[1]}\n")
        else:
            for i in range(len(boundary_nodes)):
                # Element ID (start from end of last elements and continue), type (1=2-node line)
                # tags (2, boundary_tag, 1), node IDs
                f.write(
                    f"{len(elements) + i + 1} 1 2 {boundary_tag} 1 {boundary_nodes[i]} {boundary_nodes[(i + 1) % len(boundary_nodes)]}\n")

        f.write("$EndElements\n")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    mesh = UnitSquareMesh(2, 2) # 2x2 mesh
    uu, u_true, F, pde_params = poisson2d_fsin_b0(mesh) # solve Poisson equation
    plot_test_poisson_mesh(mesh, uu) # plot solution and mesh

    fname = "Testmesh.msh"
    create_Testmesh(fname) # create mesh file
    mesh = Mesh(fname) # load mesh from file
    uu, u_true, F, pde_params = poisson2d_fsin_b0(mesh)
    plot_test_poisson_mesh(mesh, uu)

    # mesh, u = poisson_mesh_and_solve(fname)
