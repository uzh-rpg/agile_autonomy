import matplotlib.cm as cm
import os

import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd

import argparse

from utils import mesh_utils

########################################################
# Define data root directory, find subdirectories
########################################################

parser = argparse.ArgumentParser(description='Visualize Trajectory Labels')
parser.add_argument('--data_dir',
                    help='Path to data', required=True)
parser.add_argument('--data_dir_2',
                    help='Path to data 2. If given will plot them both', required=False)
parser.add_argument('--start_idx',
                    help='Start plotting at this trajectory', required=False)
parser.add_argument('--time_steps',
                    help='How many timesteps to plot', required=False)
parser.add_argument('--max_states',
                    help='For each plotted trajectory, how many states to plot', required=False)
parser.add_argument('--max_traj_to_plot',
                    help='How many trajectories to plot', required=False)
parser.add_argument('--pc_cutoff_z',
                    help='Crop pointcloud in z-axis', required=False)

args = parser.parse_args()
data_dir = args.data_dir
data_dir_2 = args.data_dir_2
start_idx = args.start_idx
num_timesteps = args.time_steps
max_states = args.max_states
if data_dir_2 is not None:
    print("Two data directories specified, will compare rollouts.")
    compare = True
else:
    compare = False

########################################################
# Visualization Parameters
########################################################
default_folder = 0  # which rollout to visualize if root directory is specified
if start_idx is not None:
    # within folder, where to start plotting
    default_start_idx = int(start_idx)
else:
    default_start_idx = 0  # within folder, where to start plotting

if num_timesteps is not None:
    default_length = int(num_timesteps)  # how many planning steps to plot
else:
    default_length = 1  # how many planning steps to plot

if max_states is not None:
    max_states = int(max_states)  # how many planning steps to plot
else:
    max_states = 100  # how many planning steps to plot

max_traj_to_plot = args.max_traj_to_plot
if max_traj_to_plot is not None:
    max_traj_to_plot = int(max_traj_to_plot)
else:
    max_traj_to_plot = 2

pc_cutoff_z = args.pc_cutoff_z
if pc_cutoff_z is not None:
    pc_cutoff_z = float(pc_cutoff_z)
else:
    pc_cutoff_z = 5.0


visualize_pointcloud = True
visualize_gate = False
visualize_poles = False
visualize_wall = False
visualize_floor = False
visualize_drone = True
step = 1

########################################################
# possible data_dir are either the root data directory or a specific rollout folder
if not os.path.isdir(data_dir):
    print("Specified directory does not exist!")
    exit(0)

subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
if any("rollout_" in os.path.basename(os.path.normpath(subfolder)) for subfolder in subfolders):
    print("Root data directory specified, will visualize folder with index %d." %
          default_folder)
    rollout_dir = subfolders[default_folder]
else:
    # "rollout_" in os.path.basename(os.path.normpath(data_dir)):
    print("Specific rollout specified")
    rollout_dir = data_dir
    if compare:
        rollout_dir_2 = data_dir_2

print("Visualizing rollout: %s" % rollout_dir)

viz_list = []

########################################################
# Load pointcloud
########################################################
if visualize_pointcloud:
    print("Loading pointcloud...")
    pointcloud = o3d.io.read_point_cloud(
        rollout_dir + "/pointcloud-unity.ply")
    if compare and len(pointcloud.points) == 0:
        pointcloud = o3d.io.read_point_cloud(
            rollout_dir_2 + "/pointcloud-unity.ply")

    # pointcloud.paint_uniform_color([0.3, 0.3, 0.3])
    # crop pointcloud
    # pointcloud = o3d.geometry.crop_point_cloud(
    #     pointcloud, [-1, -1, -1], [1, 0.6, 1])
    pts = np.asarray(pointcloud.points)
    # import pdb
    # pdb.set_trace()
    pts_cropped = pts[pts[:, 2] < pc_cutoff_z][:]
    pointcloud.points = o3d.utility.Vector3dVector(pts_cropped)

    obstacles_numpy = np.asarray(pointcloud.points)
    viz_list.append(pointcloud)

########################################################
# Load gate mesh, create mesh for ground plane
########################################################
# load gate mesh
if visualize_gate:
    print("Loading gate...")
    gate_mesh = o3d.io.read_triangle_mesh("meshes/gate_converted.stl")
    gate_mesh_transform = np.asarray(
        [[1.0, 0.0, 0.0, 4.0],
         [0.0, 0.0, -1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 1.0]])
    gate_mesh.transform(gate_mesh_transform)
    gate_mesh.compute_vertex_normals()
    gate_mesh.paint_uniform_color([0.9, 0.1, 0.1])
    viz_list.append(gate_mesh)
    gate_mesh_2 = o3d.io.read_triangle_mesh("meshes/gate_converted.stl")
    gate_mesh_transform_2 = np.asarray(
        [[1.0, 0.0, 0.0, 4.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, -1.0, 0.0, 5.0],
         [0.0, 0.0, 0.0, 1.0]])
    gate_mesh_2.transform(gate_mesh_transform_2)
    gate_mesh_2.compute_vertex_normals()
    gate_mesh_2.paint_uniform_color([0.9, 0.1, 0.1])
    viz_list.append(gate_mesh_2)

if visualize_poles:
    print("Loading poles...")
    poles_fname = rollout_dir + "/pole_positions.csv"
    print(poles_fname)
    df_poles = pd.read_csv(poles_fname)

    for index, row in df_poles.iterrows():
        print(row['pos_x'], row['pos_y'], row['pos_z'])
        pole_mesh = o3d.geometry.TriangleMesh.create_cylinder(0.3, 10.0)
        pole_mesh_transform = np.asarray(
            [[1.0, 0.0, 0.0, row['pos_x']],
             [0.0, 1.0, 0.0, row['pos_y']],
             [0.0, 0.0, 1.0, row['pos_z']],
             [0.0, 0.0, 0.0, 1.0]])
        pole_mesh.transform(pole_mesh_transform)
        pole_mesh.compute_vertex_normals()
        pole_mesh.paint_uniform_color([0.1, 0.1, 0.9])
        viz_list.append(pole_mesh)

if visualize_wall:
    print("Loading double slit...")
    double_slit_mesh = o3d.io.read_triangle_mesh("meshes/double_slit_v2.stl")
    double_slit_mesh_transform = np.asarray(
        [[0.0, -1.0, 0.0, 4.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 4.0],
         [0.0, 0.0, 0.0, 1.0]])
    double_slit_mesh.transform(double_slit_mesh_transform)
    double_slit_mesh.compute_vertex_normals()
    double_slit_mesh.paint_uniform_color([0.9, 0.1, 0.1])
    viz_list.append(double_slit_mesh)

if visualize_floor:
    print("Loading ground plane...")
    floor_mesh = o3d.geometry.TriangleMesh.create_box(
        width=20.0, height=10.0, depth=0.1)
    floor_mesh_transform = np.asarray(
        [[1.0, 0.0, 0.0, -10.0],
         [0.0, 1.0, 0.0, -5.0],
         [0.0, 0.0, 1.0, 0.2],
         [0.0, 0.0, 0.0, 1.0]])
    floor_mesh.transform(floor_mesh_transform)
    floor_mesh.compute_vertex_normals()
    floor_mesh.paint_uniform_color([0.3, 0.3, 0.3])
    viz_list.append(floor_mesh)

########################################################
# Load trajectories
########################################################
print("Loading trajectories...")
# colormap for visualization
cmap = cm.get_cmap('inferno')
cmap2 = cm.get_cmap('winter')

for timestep in np.arange(default_start_idx, default_start_idx + default_length, step=step):
    traj_fname = rollout_dir + "/trajectories/trajectories_wf_" + \
        '{:08d}'.format(timestep) + ".csv"
    if compare:
        traj_fname_2 = rollout_dir_2 + "/trajectories/trajectories_wf_" + \
            '{:08d}'.format(timestep) + ".csv"
    print(traj_fname)
    try:
        df_trajectories = pd.read_csv(traj_fname)
        if compare:
            df_trajectories_2 = pd.read_csv(traj_fname_2)
    except:
        continue

    # set rel cost if it does not exist
    try:
        rel_cost = df_trajectories['rel_cost'].values
    except:
        df_trajectories['rel_cost'] = np.zeros((df_trajectories.shape[0], 1))

    # get trajectory with highest cost
    highest_cost = 0.0
    if (max_traj_to_plot < len(df_trajectories)):
        highest_cost = df_trajectories.iloc[max_traj_to_plot, -1]
    else:
        highest_cost = df_trajectories['rel_cost'].max()

    num_traj_to_plot = min(max_traj_to_plot, len(df_trajectories))
    if compare:
        num_traj_to_plot = min(num_traj_to_plot, len(df_trajectories_2))
    print("Plotting %d trajectories..." % num_traj_to_plot)
    quad_colors = np.array([[0,0,255], [255,0,0], [0,255,0]]) / 255
    for i in range(num_traj_to_plot):
        # iterate over trajectories
        rel_cost = df_trajectories['rel_cost'].values[i]
        x_pos = np.expand_dims(df_trajectories['pos_x_0'].values[i], axis=0)
        y_pos = np.expand_dims(df_trajectories['pos_y_0'].values[i], axis=0)
        z_pos = np.expand_dims(df_trajectories['pos_z_0'].values[i], axis=0)

        # load mesh & add to viz_list
        if visualize_drone: # and i == 0:
            att_w = np.expand_dims(df_trajectories['q_w_0'].values[i], axis=0)
            att_x = np.expand_dims(df_trajectories['q_x_0'].values[i], axis=0)
            att_y = np.expand_dims(df_trajectories['q_y_0'].values[i], axis=0)
            att_z = np.expand_dims(df_trajectories['q_z_0'].values[i], axis=0)
            quad_mesh = mesh_utils.load_drone_mesh(np.concatenate((x_pos, y_pos, z_pos), axis=0),
                                                   np.concatenate((att_w, att_x, att_y, att_z), axis=0))
            #rgba = cmap(1.0 - rel_cost / highest_cost)
            rgba = quad_colors[i]
            quad_mesh.paint_uniform_color([rgba[0], rgba[1], rgba[2]])
            viz_list.append(quad_mesh)

        edges = []
        colors = []
        # max_states = 100
        for j in range(1, max_states):
            # iterate over states in trajectory
            #try:
            curr_x_pos = np.expand_dims(
                df_trajectories['pos_x_{}'.format(j)].values[i], axis=0) / 3.
            curr_y_pos = np.expand_dims(
                df_trajectories['pos_y_{}'.format(j)].values[i], axis=0) / 3.
            curr_z_pos = np.expand_dims(
                df_trajectories['pos_z_{}'.format(j)].values[i], axis=0) / 3.
            x_pos = np.concatenate((x_pos, curr_x_pos), axis=0)
            y_pos = np.concatenate((y_pos, curr_y_pos), axis=0)
            z_pos = np.concatenate((z_pos, curr_z_pos), axis=0)
            edges.append([j - 1, j])
            if visualize_drone and j % 1 == 0:
                att_w = np.expand_dims(
                    df_trajectories['q_w_{}'.format(j)].values[i], axis=0)
                att_x = np.expand_dims(
                    df_trajectories['q_x_{}'.format(j)].values[i], axis=0)
                att_y = np.expand_dims(
                    df_trajectories['q_y_{}'.format(j)].values[i], axis=0)
                att_z = np.expand_dims(
                df_trajectories['q_z_{}'.format(j)].values[i], axis=0)
                quad_mesh = mesh_utils.load_drone_mesh(np.concatenate((curr_x_pos, curr_y_pos, curr_z_pos), axis=0),
                                                       np.concatenate((att_w, att_x, att_y, att_z), axis=0))
                rgba = quad_colors[i]
                quad_mesh.paint_uniform_color([rgba[0], rgba[1], rgba[2]])
                viz_list.append(quad_mesh)
            #except:
            #    break

        # [n_states, 3] array of positions
        xyz = np.concatenate((np.expand_dims(np.reshape(x_pos, -1), axis=1), np.expand_dims(
            np.reshape(y_pos, -1), axis=1), np.expand_dims(np.reshape(z_pos, -1), axis=1)), axis=1)
        o3d_traj = o3d.geometry.PointCloud()
        o3d_traj.points = o3d.utility.Vector3dVector(xyz)

        if compare:
            x_pos = np.expand_dims(
                df_trajectories_2['pos_x_0'].values[i], axis=0)
            y_pos = np.expand_dims(
                df_trajectories_2['pos_y_0'].values[i], axis=0)
            z_pos = np.expand_dims(
                df_trajectories_2['pos_z_0'].values[i], axis=0)
            #rel_cost = df_trajectories_2['rel_cost'].values[i]
            edges = []
            colors = []
            # max_states = 100
            for j in range(1, max_states):
                try:
                    x_pos = np.concatenate((x_pos, np.expand_dims(
                        df_trajectories_2['pos_x_{}'.format(j)].values[i], axis=0)), axis=0)
                    y_pos = np.concatenate((y_pos, np.expand_dims(
                        df_trajectories_2['pos_y_{}'.format(j)].values[i], axis=0)), axis=0)
                    z_pos = np.concatenate((z_pos, np.expand_dims(
                        df_trajectories_2['pos_z_{}'.format(j)].values[i], axis=0)), axis=0)
                    edges.append([j - 1, j])
                except:
                    break

            xyz_2 = np.concatenate((np.expand_dims(np.reshape(x_pos, -1), axis=1), np.expand_dims(
                np.reshape(y_pos, -1), axis=1), np.expand_dims(np.reshape(z_pos, -1), axis=1)), axis=1)
            o3d_traj_2 = o3d.geometry.PointCloud()
            o3d_traj_2.points = o3d.utility.Vector3dVector(xyz_2)
            # colorize trajectory according to cost. value passed to colormap is [0, 1]
            rgba = (1.0, 0.0, 0.0, 1.0)
            # rgba = cmap(1.0 - rel_cost / highest_cost)
            o3d_traj.paint_uniform_color([rgba[0], rgba[1], rgba[2]])
            viz_list.append(o3d_traj)
            colors = [[rgba[0], rgba[1], rgba[2]] for i in range(len(edges))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(xyz),
                lines=o3d.utility.Vector2iVector(edges),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            viz_list.append(line_set)

            rgba = (0.0, 0.0, 1.0, 1.0)
            o3d_traj_2.paint_uniform_color([rgba[0], rgba[1], rgba[2]])
            viz_list.append(o3d_traj_2)
            # vis.add_geometry(o3d_traj)
            colors = [[rgba[0], rgba[1], rgba[2]] for i in range(len(edges))]
            line_set_2 = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(xyz_2),
                lines=o3d.utility.Vector2iVector(edges),
            )
            line_set_2.colors = o3d.utility.Vector3dVector(colors)
            viz_list.append(line_set_2)
        else:
            # colorize trajectory according to cost. value passed to colormap is [0, 1]
            rgba = cmap(1.0 - rel_cost / highest_cost)
            o3d_traj.paint_uniform_color([rgba[0], rgba[1], rgba[2]])
            viz_list.append(o3d_traj)
            colors = [[rgba[0], rgba[1], rgba[2]] for i in range(len(edges))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(xyz),
                lines=o3d.utility.Vector2iVector(edges),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            #viz_list.append(line_set)

# all trajectories are loaded, visualize
o3d.visualization.draw_geometries(viz_list)
# remove latest
for j in range(2 + 2 * compare):
    viz_list.pop(-1)
# viz_list.remove(o3d_traj)
# viz_list.remove(line_set)
