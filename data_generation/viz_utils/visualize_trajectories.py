import matplotlib.cm as cm
import os

import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd

import argparse

from scipy.spatial.transform import Rotation as R

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
visualize_odometry = False
visualize_ideal_reference = False
visualize_trajectories = True
visualize_start_goal = True
crop_xy = False
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

if visualize_odometry:
    odom_fname = rollout_dir + "/odometry.csv"
    df_odometry = pd.read_csv(odom_fname)

    edges = []
    orig_time = df_odometry['time_from_start'].to_numpy()
    new_time = np.linspace(0, orig_time[-1], 1000)
    x_pos = df_odometry['pos_x']
    y_pos = df_odometry['pos_y']
    z_pos = df_odometry['pos_z']
    q_w = df_odometry['q_w']
    q_x = df_odometry['q_x']
    q_y = df_odometry['q_y']
    q_z = df_odometry['q_z']

    odom_xyz = np.concatenate([x_pos, y_pos, z_pos], axis=1)
    odom_att = np.concatenate([q_w, q_x, q_y, q_z], axis=1)

    for odom_idx in range(odom_xyz.shape[0]):
        hack_x = 0.0
        hack_y = 0.0

        odom_mesh = o3d.geometry.TriangleMesh.create_cylinder(0.1, 0.04)

        rot_body = R.from_quat([odom_att[odom_idx, 1],
                                odom_att[odom_idx, 2],
                                odom_att[odom_idx, 3],
                                odom_att[odom_idx, 0]])
        R_odom = rot_body.as_matrix()
        odom_mesh_transform = np.asarray(
            [[R_odom[0, 0], R_odom[0, 1], R_odom[0, 2], odom_xyz[odom_idx, 0] + hack_x],
             [R_odom[1, 0], R_odom[1, 1], R_odom[1, 2], odom_xyz[odom_idx, 1] + hack_y],
             [R_odom[2, 0], R_odom[2, 1], R_odom[2, 2], odom_xyz[odom_idx, 2]],
             [0.0, 0.0, 0.0, 1.0]])
        odom_mesh.transform(odom_mesh_transform)
        odom_mesh.compute_vertex_normals()
        odom_mesh.paint_uniform_color([0.0, 0.8, 0.0])
        viz_list.append(odom_mesh)

if visualize_ideal_reference:
    ref_fname = rollout_dir + "/reference_trajectory.csv"
    df_reference = pd.read_csv(ref_fname)

    edges = []
    x_pos = df_reference['pos_x'].to_numpy()[:, np.newaxis]
    y_pos = df_reference['pos_y'].to_numpy()[:, np.newaxis]
    z_pos = df_reference['pos_z'].to_numpy()[:, np.newaxis]
    q_w = df_reference['q_w'].to_numpy()[:, np.newaxis]
    q_x = df_reference['q_x'].to_numpy()[:, np.newaxis]
    q_y = df_reference['q_y'].to_numpy()[:, np.newaxis]
    q_z = df_reference['q_z'].to_numpy()[:, np.newaxis]

    ref_xyz = np.concatenate([x_pos, y_pos, z_pos], axis=1)
    ref_att = np.concatenate([q_w, q_x, q_y, q_z], axis=1)

    for ref_idx in range(ref_xyz.shape[0]):
        if not ref_idx % 1 == 0:
            continue
        pole_mesh = o3d.geometry.TriangleMesh.create_cylinder(0.1, 0.04)

        rot_body = R.from_quat([ref_att[ref_idx, 1],
                                ref_att[ref_idx, 2],
                                ref_att[ref_idx, 3],
                                ref_att[ref_idx, 0]])
        R_ref = rot_body.as_matrix()
        pole_mesh_transform = np.asarray(
            [[R_ref[0, 0], R_ref[0, 1], R_ref[0, 2], ref_xyz[ref_idx, 0]],
             [R_ref[1, 0], R_ref[1, 1], R_ref[1, 2], ref_xyz[ref_idx, 1]],
             [R_ref[2, 0], R_ref[2, 1], R_ref[2, 2], ref_xyz[ref_idx, 2]],
             [0.0, 0.0, 0.0, 1.0]])
        pole_mesh.transform(pole_mesh_transform)
        pole_mesh.compute_vertex_normals()
        pole_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        viz_list.append(pole_mesh)

if visualize_start_goal:
    ref_fname = rollout_dir + "/reference_trajectory.csv"
    df_reference = pd.read_csv(ref_fname)

    x_pos = df_reference['pos_x'].to_numpy()[:, np.newaxis]
    y_pos = df_reference['pos_y'].to_numpy()[:, np.newaxis]
    z_pos = df_reference['pos_z'].to_numpy()[:, np.newaxis]

    z_offset = 5.0

    start_pos = np.concatenate([x_pos[0, np.newaxis], y_pos[0, np.newaxis], z_offset + z_pos[0, np.newaxis]], axis=1)
    goal_pos = np.concatenate([x_pos[-1, np.newaxis], y_pos[-1, np.newaxis], z_offset + z_pos[-1, np.newaxis]], axis=1)
    start_pos = np.squeeze(start_pos)
    goal_pos = np.squeeze(goal_pos)

    start_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.5, cone_radius=0.75, cylinder_height=2.5,
                                                        cone_height=2.0, resolution=20, cylinder_split=4, cone_split=1)
    rot_body = R.from_quat([1.0,
                            0.0,
                            0.0,
                            0.0])
    R_ref = rot_body.as_matrix()
    start_mesh_transform = np.asarray(
        [[R_ref[0, 0], R_ref[0, 1], R_ref[0, 2], start_pos[0]],
         [R_ref[1, 0], R_ref[1, 1], R_ref[1, 2], start_pos[1]],
         [R_ref[2, 0], R_ref[2, 1], R_ref[2, 2], start_pos[2]],
         [0.0, 0.0, 0.0, 1.0]])
    start_mesh.transform(start_mesh_transform)
    start_mesh.compute_vertex_normals()
    start_mesh.paint_uniform_color([0.0, 1.0, 0.0])
    viz_list.append(start_mesh)

    goal_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.5, cone_radius=0.75, cylinder_height=2.5,
                                                       cone_height=2.0, resolution=20, cylinder_split=4, cone_split=1)
    goal_mesh_transform = np.asarray(
        [[R_ref[0, 0], R_ref[0, 1], R_ref[0, 2], goal_pos[0]],
         [R_ref[1, 0], R_ref[1, 1], R_ref[1, 2], goal_pos[1]],
         [R_ref[2, 0], R_ref[2, 1], R_ref[2, 2], goal_pos[2]],
         [0.0, 0.0, 0.0, 1.0]])
    goal_mesh.transform(goal_mesh_transform)
    goal_mesh.compute_vertex_normals()
    goal_mesh.paint_uniform_color([1.0, 0.0, 0.0])
    viz_list.append(goal_mesh)

########################################################
# Load trajectories
########################################################
if visualize_trajectories:
    print("Loading trajectories...")
    # colormap for visualization
    cmap = cm.get_cmap('jet')
    cmap2 = cm.get_cmap('winter')

    # keep track of the most extreme trajectories
    min_x = 999
    max_x = -999
    min_y = 999
    max_y = -999
    min_z = 999
    max_z = -999

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
            lowest_cost = df_trajectories.iloc[0, -1]
            highest_cost = df_trajectories.iloc[max_traj_to_plot, -1]
        else:
            lowest_cost = df_trajectories['rel_cost'].min()
            highest_cost = df_trajectories['rel_cost'].max()

        num_traj_to_plot = min(max_traj_to_plot, len(df_trajectories))
        if compare:
            num_traj_to_plot = min(num_traj_to_plot, len(df_trajectories_2))
        print("Plotting %d trajectories..." % num_traj_to_plot)
        for i in range(num_traj_to_plot):
            # iterate over trajectories
            rel_cost = df_trajectories['rel_cost'].values[i] - lowest_cost
            x_pos = np.expand_dims(df_trajectories['pos_x_0'].values[i], axis=0)
            y_pos = np.expand_dims(df_trajectories['pos_y_0'].values[i], axis=0)
            z_pos = np.expand_dims(df_trajectories['pos_z_0'].values[i], axis=0)

            edges = []
            colors = []
            max_thrust_exceeded = False
            for j in range(1, max_states):
                # iterate over states in trajectory
                try:
                    curr_x_pos = np.expand_dims(
                        df_trajectories['pos_x_{}'.format(j)].values[i], axis=0)
                    curr_y_pos = np.expand_dims(
                        df_trajectories['pos_y_{}'.format(j)].values[i], axis=0)
                    curr_z_pos = np.expand_dims(
                        df_trajectories['pos_z_{}'.format(j)].values[i], axis=0)
                    if df_trajectories['thrust_{}'.format(j)].values[i] > 60.0:
                        max_thrust_exceeded = True
                        break
                    x_pos = np.concatenate((x_pos, curr_x_pos), axis=0)
                    y_pos = np.concatenate((y_pos, curr_y_pos), axis=0)
                    z_pos = np.concatenate((z_pos, curr_z_pos), axis=0)
                    edges.append([j - 1, j])
                except:
                    break

            if max_thrust_exceeded:
                continue
            # [n_states, 3] array of positions
            xyz = np.concatenate((np.expand_dims(np.reshape(x_pos, -1), axis=1), np.expand_dims(
                np.reshape(y_pos, -1), axis=1), np.expand_dims(np.reshape(z_pos, -1), axis=1)), axis=1)

            min_x = np.min([min_x, np.min(xyz[:, 0])])
            max_x = np.max([max_x, np.max(xyz[:, 0])])
            min_y = np.min([min_y, np.min(xyz[:, 1])])
            max_y = np.max([max_y, np.max(xyz[:, 1])])
            min_z = np.min([min_z, np.min(xyz[:, 2])])
            max_z = np.max([max_z, np.max(xyz[:, 2])])

            min_z = -0.5

            o3d_traj = o3d.geometry.PointCloud()
            o3d_traj.points = o3d.utility.Vector3dVector(xyz)

            if compare:
                x_pos = np.expand_dims(
                    df_trajectories_2['pos_x_0'].values[i], axis=0)
                y_pos = np.expand_dims(
                    df_trajectories_2['pos_y_0'].values[i], axis=0)
                z_pos = np.expand_dims(
                    df_trajectories_2['pos_z_0'].values[i], axis=0)
                edges = []
                colors = []
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
                rgba = cmap(1.0 - rel_cost / (highest_cost - lowest_cost))
                o3d_traj.paint_uniform_color([rgba[0], rgba[1], rgba[2]])
                viz_list.append(o3d_traj)
                colors = [[rgba[0], rgba[1], rgba[2]] for i in range(len(edges))]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(xyz),
                    lines=o3d.utility.Vector2iVector(edges),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                viz_list.append(line_set)

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

    # crop pointcloud
    pts = np.asarray(pointcloud.points)

    pc_cutoff_z_min = -1.5
    pts_cropped = pts[pts[:, 2] < pc_cutoff_z][:]
    pts_cropped = pts_cropped[pts_cropped[:, 2] > pc_cutoff_z_min][:]
    if crop_xy:
        padding = 5.0
        min_x = min_x - padding
        max_x = max_x + padding
        min_y = min_y - padding
        max_y = max_y + padding
        pts_cropped = pts_cropped[np.logical_and(pts_cropped[:, 0] < max_x, pts_cropped[:, 0] > min_x)][:]
        pts_cropped = pts_cropped[np.logical_and(pts_cropped[:, 1] < max_y, pts_cropped[:, 1] > min_y)][:]

    pts_cropped = pts_cropped[np.logical_not(np.logical_and(np.logical_and(
        np.logical_and(pts_cropped[:, 0] < -19.5, pts_cropped[:, 0] > -20.5),
        np.logical_and(pts_cropped[:, 1] < 15.5, pts_cropped[:, 1] > 14.5)),
        np.logical_and(pts_cropped[:, 2] < 5.0, pts_cropped[:, 2] > 2.0)))][:]

    pointcloud.points = o3d.utility.Vector3dVector(pts_cropped)
    point_colors = np.zeros_like(pts_cropped)
    cmap = cm.get_cmap('inferno')
    z_0_1 = (pts_cropped[:, 2] - np.min(pts_cropped[:, 2])) / (np.max(pts_cropped[:, 2]) - np.min(pts_cropped[:, 2]))
    rgba = cmap(z_0_1)
    print(rgba.shape)

    pointcloud.colors = o3d.utility.Vector3dVector(rgba[:, :3])

    obstacles_numpy = np.asarray(pointcloud.points)
    viz_list.append(pointcloud)

########################################################
# View point stuff
########################################################
viewpoint_params = '/tmp/viewpoint.json'
viewpoint_params2 = 'plot_viewpoint.json'

# this one is used to illustrate global planning vs no global planning
vis = o3d.visualization.Visualizer()
vis.create_window()
ctr = vis.get_view_control()
param = o3d.io.read_pinhole_camera_parameters(viewpoint_params2)

for viz_item in range(len(viz_list)):
    vis.add_geometry(viz_list[viz_item])
ctr.convert_from_pinhole_camera_parameters(param)
vis.run()  # user changes the view and press "q" to terminate
param = vis.get_view_control().convert_to_pinhole_camera_parameters()
o3d.io.write_pinhole_camera_parameters(viewpoint_params, param)
vis.destroy_window()
