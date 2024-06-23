import numpy as np
import torch
from torch_geometric.utils import is_undirected, to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from adjustText import adjust_text
from scipy.spatial import ConvexHull

@torch.no_grad()
def loss_histogram(out, x_phys):
    # Compute the element-wise absolute differences
    differences = torch.abs(out - x_phys)

    # Flatten the differences to a 1D tensor
    differences_flat = differences.view(-1)

    # Convert to a NumPy array for plotting
    differences_numpy = differences_flat.numpy()

    # Plot a histogram
    plt.hist(differences_numpy, bins=30)
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Contributions to the Loss')
    plt.show()

def plot_training_evol(y_list, title=None, ax=None, plot_show=False, batch_loss_list=None, batches_per_epoch=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(y_list)
    if batch_loss_list is not None and batches_per_epoch is not None:
        epoch_scale_indices = list(np.linspace(1/batches_per_epoch, len(y_list) - 1, len(batch_loss_list)))
        ax.plot(epoch_scale_indices, batch_loss_list, color='red', label='Batch Loss')

    ax.set_xlabel('Epoch/Batch update')
    ax.set_ylabel('Loss')
    if title is not None:
        ax.set_title(title)
    if plot_show:
        plt.show()
    return ax


def plot_mesh_evol(mesh_list, title=None, ax=None, plot_show=False):
    """
    Plot the evolution of mesh node positions over time.

    :param mesh_list: A list of tensors, each tensor is a mesh at time t.
    :param title: Optional title for the plot.
    :return: The matplotlib figure object.
    """
    if ax is None:
        fig, ax = plt.subplots()

    np_mesh_tensor = torch.stack(mesh_list).to('cpu').detach().numpy()
    num_time_steps = np_mesh_tensor.shape[0]
    for i in range(np_mesh_tensor.shape[1]):  # Assuming shape is [time, nodes, dimensions]
        x_coords = np_mesh_tensor[:, i]#, 0]
        time_steps = np.arange(num_time_steps)
        ax.plot(x_coords, time_steps, label=f'Node {i} X' if i == 0 else None)

    ax.set_xlabel('Node Positions')
    ax.set_ylabel('Time Steps')

    if title is not None:
        ax.set_title(title)

    if plot_show:
        plt.show()

    return ax


def vizualise_grid(mesh_tensor, opt=None):
    np_mesh_tensor = mesh_tensor.cpu().detach().numpy()

    fig = plt.figure()
    if np_mesh_tensor.shape[1] == 1:
        plt.scatter(np_mesh_tensor, np.zeros(np_mesh_tensor.shape[0]))
    elif np_mesh_tensor.shape[1] == 2:
        plt.scatter(np_mesh_tensor[:, 0], np_mesh_tensor[:, 1])
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np_mesh_tensor[:, 0], np_mesh_tensor[:, 1], np_mesh_tensor[:, 2])

    if opt is not None:
        if opt['show_plots']:
            plt.show()
    return fig

def vizualise_grid_with_edges(mesh_tensor, edges, ax=None, opt=None, boundary_nodes=None, directed_edges=False, node_labels=False,
                              node_boundary_map=None, corner_nodes=None, edge_weights=None, width=0.2):
    np_mesh_tensor = mesh_tensor.cpu().detach().numpy()

    if edge_weights is not None:
        edge_weights = np.array(edge_weights.to('cpu').detach())  # Convert to a numpy array if not already
        norm = plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
        cmap = plt.get_cmap('viridis')  # Choose a colormap

    if ax is None:
        fig, ax = plt.subplots()

    if np_mesh_tensor.shape[1] == 1:
        plt.scatter(np_mesh_tensor, np.zeros(np_mesh_tensor.shape[0]))
        for edge in edges.T:
            if directed_edges:
                ax.plot(np_mesh_tensor[edge], [0, 0], '->')
            else:
                ax.plot(np_mesh_tensor[edge], [0, 0])
    elif np_mesh_tensor.shape[1] == 2:
        ax.scatter(np_mesh_tensor[:, 0], np_mesh_tensor[:, 1])
        for i, edge in enumerate(edges.T):
            start = np_mesh_tensor[edge[0]]
            end = np_mesh_tensor[edge[1]]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if directed_edges:
                if edge_weights is not None:
                    if torch.isclose(edge_weights[i], torch.tensor([1.]), atol=1e-6):
                        col = 'r'
                        ax.arrow(start[0], start[1], dx, dy, head_width=0.05, head_length=0.05, color=col, linewidth=width)
                    elif torch.isclose(edge_weights[i], torch.tensor([0.]), atol=1e-6):
                        col = 'b'
                        ax.arrow(start[0], start[1], dx, dy, head_width=0.05, head_length=0.05, color=col, linewidth=width)
                    else:
                        ax.arrow(start[0], start[1], dx, dy, head_width=0.05, head_length=0.05)#, fc='k', ec='k')
                else:
                    ax.arrow(start[0], start[1], dx, dy, head_width=0.05, head_length=0.05)#, fc='k', ec='k')
                # plt.plot(np_mesh_tensor[edge, 0], np_mesh_tensor[edge, 1], '->', markersize=10, markeredgewidth=2)
            else:
                offset = 0.01
                # plt.plot(np_mesh_tensor[edge, 0], np_mesh_tensor[edge, 1])
                col = cmap(norm(edge_weights[i]))  # Map weight to color
                # plt.plot([start[0], end[0]], [start[1], end[1]], color=col)

                # shift 1 line width down/left
                if edge[0] >= edge[1]:
                    start = start - offset #np.array([0., -offset])
                    end = end - offset#+ np.array([0., -offset])
                    # plt.plot([start[0], end[0]], [start[1], end[1]], color='r')
                # shift line up/right
                else:
                    start = start + offset #np.array([0., offset])
                    end = end + offset #np.array([0., offset])
                    # plt.plot([start[0], end[0]], [start[1], end[1]], color='g')
                ax.plot([start[0], end[0]], [start[1], end[1]], color=col, linewidth=width)
                # plt.arrow(start[0], start[1], dx, dy, head_width=0.05, head_length=0.05, color=col)#, linewidth=width)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np_mesh_tensor[:, 0], np_mesh_tensor[:, 1], np_mesh_tensor[:, 2])
        for edge in edges.T:
            if directed_edges:
                ax.plot(np_mesh_tensor[edge, 0], np_mesh_tensor[edge, 1], np_mesh_tensor[edge, 2], '->')
            else:
                ax.plot(np_mesh_tensor[edge, 0], np_mesh_tensor[edge, 1], np_mesh_tensor[edge, 2])

    if boundary_nodes is not None:
        #change boundary nodes to red squares
        if np_mesh_tensor.shape[1] == 1:
            ax.scatter(np_mesh_tensor[boundary_nodes], np.zeros(len(boundary_nodes)), marker='s', c='r')
        elif np_mesh_tensor.shape[1] == 2:
            ax.scatter(np_mesh_tensor[boundary_nodes, 0], np_mesh_tensor[boundary_nodes, 1], marker='s', c='r')
        else:
            ax.scatter(np_mesh_tensor[boundary_nodes, 0], np_mesh_tensor[boundary_nodes, 1], np_mesh_tensor[boundary_nodes, 2], marker='s', c='r')

    #this is really for analysis of the derformed mesh crossing
    if node_labels:
        texts = []  # This will store the text objects for adjustText
        CVH_breaches = []

        # for i in range(np_mesh_tensor.shape[0]):
        #     plt.annotate(i, (np_mesh_tensor[i, 0], np_mesh_tensor[i, 1]))
        # Get the center and radius of the data
        center = np.array([0.3, 0.3]) #np.mean(np_mesh_tensor, axis=0)
        radius = 0.5#np.max(np.linalg.norm(np_mesh_tensor - center, axis=1))

        # Adjust the plot limits to accommodate the labels
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min - 0.2 * radius, x_max + 0.2 * radius)
        ax.set_ylim(y_min - 0.2 * radius, y_max + 0.2 * radius)

        focus_nodes = 16
        for i, point in enumerate(np_mesh_tensor):
            if i == focus_nodes:
                break
            # Compute the direction from the center to the point
            direction = point - center
            direction /= np.linalg.norm(direction)

            # Define an offset for label from the point
            offset = 1.2 + i / focus_nodes # Adjust this as per your need

            # New label position
            label_pos = center + offset * radius * direction

            # Annotate the point with a label and an arrow
            head_length = 0.1
            text = ax.annotate(str(i),
                         xy=point[:2],
                         xytext=label_pos[:2],
                         arrowprops=dict(arrowstyle="-|>",
                                         shrinkA=0, shrinkB=0))

            if i in corner_nodes[0]:
                node_type = f"{i} (corner)"
            elif i in node_boundary_map.keys():
                node_type = f"boundary {node_boundary_map[i]}"
            else:
                node_type = "interior"

            neighbors = edges[0][edges[1] == i]
            neighbor_positions = mesh_tensor[neighbors]
            if is_outside_convex_hull(point, neighbor_positions.detach().numpy()):
                in_CVH = "outside"
                CVH_breaches.append(i)
            else:
                in_CVH = "inside"

            print(f"point: {i}, pos: {point[:2]}, type: {node_type}, {in_CVH} CVH")

            texts.append(text)
        # adjust_text(texts)  # Adjust the labels to avoid overlaps
        adjust_text(texts, force_points=0.3, force_texts=5)#1.5), expand_points=(1, 1), expand_texts=(1.5, 1.5))#, lim=500)

        #test if nodes are outside convex hulls of neighbors
        nodes_against_neighbors(mesh_tensor, edges)

    if opt is not None:
        if opt['show_plots']:
            plt.show()

    if node_labels:
        for i, point in enumerate(np_mesh_tensor):
            # if i == focus_nodes:
            #     break

            if i in corner_nodes[0]:
                node_type = f"{i} (corner)"
            elif i in node_boundary_map.keys():
                node_type = f"boundary {node_boundary_map[i]}"
            else:
                node_type = "interior"

            neighbors = edges[0][edges[1] == i]
            neighbor_positions = mesh_tensor[neighbors]

            cvh_breach = i in CVH_breaches

            #nb this will also plot the convex hull of the neighbors
            if is_outside_convex_hull(point, neighbor_positions.detach().numpy(), plot_cvx_hull=cvh_breach):
                in_CVH = "outside"
            else:
                in_CVH = "inside"

            print(f"point: {i}, pos: {point[:2]}, type: {node_type}, {in_CVH} CVH")

    return ax


def plot2d(graph, compORphys="comp"):
    # Convert PyG graph to NetworkX graph
    G = to_networkx(graph, to_undirected=True)

    # Get node positions from the coordinates attribute in the PyG graph
    if compORphys == "comp":
        x = graph.x_comp
    elif compORphys == "phys":
        x = graph.x_phys
    else:
        x = graph.x

    positions = {i: x[i].tolist() for i in range(x.shape[0])}

    # Create figure and axes
    fig, ax = plt.subplots()

    # Draw the graph
    nx.draw(G, pos=positions, with_labels=True, ax=ax)

    # Show the plot
    plt.show()


def plot_3d_pyg_graph_interactive(graph, compORphys="comp"):
    if compORphys == "comp":
        x = graph.x_comp
    elif compORphys == "phys":
        x = graph.x_phys
    else:
        x = graph.x

    # Create a trace for the nodes
    trace = go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=x[:, 2], # set color to third coordinate
            colorscale='Viridis',
            opacity=0.8
        )
    )

    # Create a trace for the edges
    lines = []
    for edge in graph.edge_index.t():
        x0, x1 = x[edge, 0]
        y0, y1 = x[edge, 1]
        z0, z1 = x[edge, 2]
        lines.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines'))

    data = [trace] + lines
    fig = go.Figure(data=data)

    # Render the plot
    fig.show()

    # Save the figure in HTML
    fig.write_html("3DCube.html")

def debug_plots(data, opt):
    edge_index0 = data.edge_index
    vizualise_grid_with_edges(data.x_comp, edge_index0, opt)
    vizualise_grid_with_edges(data.x_comp, edge_index0[:, data.to_boundary_edge_mask], opt, directed_edges=True)
    vizualise_grid_with_edges(data.x_comp, edge_index0[:, data.to_corner_nodes_mask], opt, directed_edges=True)
    vizualise_grid_with_edges(data.x_comp, edge_index0[:, data.diff_boundary_edges_mask], opt, directed_edges=True)
    mask = ~data.to_boundary_edge_mask * ~data.to_corner_nodes_mask * ~data.diff_boundary_edges_mask
    vizualise_grid_with_edges(data.x_comp, edge_index0[:, mask], opt, directed_edges=True)
    vizualise_grid_with_edges(data.x_comp, edge_index0[:, ~mask], opt, directed_edges=True)
    plot2d(graph=data)


def is_outside_convex_hull(node, neighbors, plot_cvx_hull=False):
    # If there's only one neighbor, compare directly
    if len(neighbors) == 1:
        return np.linalg.norm(node - neighbors[0]) > 0.0001

    # If there are two neighbors, compute the centroid
    if len(neighbors) == 2:
        centroid = np.mean(neighbors, axis=0)
        avg_distance = np.mean([np.linalg.norm(neighbor - centroid) for neighbor in neighbors])
        return np.linalg.norm(node - centroid) > avg_distance

    # For 3 or more neighbors, use ConvexHull
    try:
        hull = ConvexHull(neighbors)
        if plot_cvx_hull:
            # Plot the convex hull
            plt.plot(neighbors[:, 0], neighbors[:, 1], 'o')  # Plot neighbors
            for simplex in hull.simplices:
                plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'g-')  # Hull edges
            plt.plot(node[0], node[1], 'ro')  # Plot node
            plt.show()
        return not np.all(hull.equations @ np.append(node, 1) <= 0)  # <= 0 if inside the convex hull
    except:
        return True  # Default to outside if any error occurs


def nodes_against_neighbors(node_positions, edge_index):
    """Test all nodes to see if any node is outside the convex hull of its neighbors."""
    outside_nodes = []
    for node_index, position in enumerate(node_positions):
        # Get neighboring nodes for the current node
        neighbors = edge_index[1][edge_index[0] == node_index]
        neighbor_positions = node_positions[neighbors]
        if is_outside_convex_hull(position.detach().numpy(), neighbor_positions.detach().numpy()):
            outside_nodes.append(node_index)

    print(f"Outside nodes: {outside_nodes}")

    return outside_nodes


if __name__ == "__main__":
    # Example usage:
    node_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    # edge_index = np.array([[0, 0, 1, 2, 2], [1, 3, 2, 3, 4]])
    edge_index = np.array([
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
    ])
    vizualise_grid_with_edges(torch.from_numpy(node_positions), torch.from_numpy(edge_index), opt={'show_plots': True})
    print(nodes_against_neighbors(node_positions, edge_index))  # This should print [4]
