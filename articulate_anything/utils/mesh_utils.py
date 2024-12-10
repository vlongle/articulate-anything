import numpy as np
import plotly.graph_objects as go


def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append(
                    [int(x.split('/')[0]) - 1 for x in line.split()[1:4]])

    return np.array(vertices), np.array(faces)


def visualize_mesh(vertices, faces):
    x, y, z = vertices.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                     color='lightblue', opacity=0.50)

    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


def transform_vertices(vertices, transform):
    return np.dot(vertices, transform[:3, :3].T) + transform[:3, 3]


def get_aabb(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    return min_coords, max_coords


def add_mesh(box_spec, mesh_dir):
    # partnet mobility artifact. Rotate the mesh upright
    up_axis_transform = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    vertices, faces = load_obj(mesh_dir)

    # Center the mesh
    min_coords, max_coords = get_aabb(vertices)
    center = (min_coords + max_coords) / 2
    vertices -= center

    # Apply up-axis transform
    vertices = transform_vertices(vertices, up_axis_transform)

    # Calculate scaling factors
    mesh_min = np.min(vertices, axis=0)
    mesh_max = np.max(vertices, axis=0)
    mesh_extent = mesh_max - mesh_min

    x_scale = box_spec["dimensions"][0] / mesh_extent[0]
    y_scale = box_spec["dimensions"][1] / mesh_extent[1]
    z_scale = box_spec["dimensions"][2] / mesh_extent[2]

    # Apply scaling
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = x_scale
    scale_matrix[1, 1] = y_scale
    scale_matrix[2, 2] = z_scale
    vertices = transform_vertices(vertices, scale_matrix)

    # Translate to final position
    # transformed_vertices = vertices + box_spec["position"]
    transformed_vertices = vertices

    # Extract x, y, z coordinates and face indices
    x, y, z = (transformed_vertices[:, 0],
               transformed_vertices[:, 1],
               transformed_vertices[:, 2])
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    return vertices, faces
