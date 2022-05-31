import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
try:
    from quadrotor_msgs.msg import TrajectoryPoint
    from quadrotor_msgs.msg import Trajectory
except:
    Trajectory=None
    TrajectoryPoint=None
from tensorflow.keras.losses import Loss
import tensorflow as tf

try:
    from pose import Pose, fromPoseMessage
except:
    from .pose import Pose, fromPoseMessage


class TrajectoryCostLoss(Loss):
    def __init__(self, ref_frame='bf', state_dim=3):
        super(TrajectoryCostLoss, self).__init__()
        self.ref_frame = ref_frame
        self.state_dim = state_dim
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def add_pointclouds(self, pointclouds):
        self.pointclouds = pointclouds

    def call(self, ids_and_states, y_pred):
        """
        ids_and_states: pointclouds ids for the batch and respective initial states
        y_pred: batch of predictions
        """
        ids = ids_and_states[0]
        states = ids_and_states[1]
        alphas = y_pred[:, :, 0]
        ids = tf.reshape(ids, (-1, 1))
        # start with simple way
        batch_size = ids.shape[0]
        traj_costs = []
        for k in range(batch_size):
            traj_costs.append(tf.stop_gradient(self._compute_traj_cost(ids[k], states[k], y_pred[k, :, 1:])))
        traj_costs = tf.stack(traj_costs)
        traj_cost_loss = 2*self.mse_loss(traj_costs, alphas)
        return traj_cost_loss

    def _compute_traj_cost(self, exp_id, state, pred_trajectories):
        num_modes = pred_trajectories.shape[0]
        costs = []
        for k in range(num_modes):
            traj_cost = tf.py_function(func=self._compute_single_traj_cost,
                                       inp=[exp_id, state, pred_trajectories[k]],
                                       Tout=tf.float32)
            traj_cost.set_shape((1,))
            costs.append(traj_cost)
        costs = tf.stack(costs)
        return costs

    def _compute_single_traj_cost(self, exp_id, state, trajectory):
        pcd_idx = int(exp_id.numpy()[0])  # matlab indexes
        pcd_tree = self.pointclouds[pcd_idx - 1]
        traj_np = trajectory.numpy()
        state = state.numpy()
        traj_len = traj_np.shape[0] // self.state_dim
        collision_threshold = 0.8
        quadrotor_size = 0.3
        # convert traj to world frame
        traj_np = np.reshape(traj_np, ((-1, traj_len)))
        traj_np = transformToWorldFrame(traj_np, start_pos=state[:3].reshape((3, 1)),
                                        start_att=state[3:].reshape((3, 3)), ref_frame=self.ref_frame)
        cost = 0.  # we will always do log
        for j in range(traj_len):
            [_, __, dists_squared] = pcd_tree.search_radius_vector_3d(traj_np[:, j],
                                                                      collision_threshold)
            if len(dists_squared) > 0:
                dist = np.sqrt(np.min(dists_squared))
                if dist < quadrotor_size:
                    # this point is in contact with the quadrotor
                    # parabolic cost with vertex in 4
                    cost += -2. / (quadrotor_size**2) * dist ** 2 + 4.
                else:
                    # linear decrease with 1 at collision_threshold
                    cost += 2 *(quadrotor_size - dist) / (collision_threshold - quadrotor_size) + 2
        # average cost
        cost = cost / traj_len
        cost = np.array(cost, dtype=np.float32).reshape((1,))
        return cost

class MixtureSpaceLoss(Loss):
    def __init__(self, T=1.5, modes=2):
        super(MixtureSpaceLoss, self).__init__()
        self.T = T
        self.space_loss = DiscretePositionLoss()
        self.modes = modes
        self.margin = 2.

    def call(self, y_true, y_pred):
        mode_losses = []
        alphas = []
        for j in range(self.modes):
            pred_len = y_pred.shape[-1]
            alpha = tf.reshape(y_pred[:, j, 0], (-1, 1))
            pred = tf.reshape(y_pred[:, j, 1:], (-1, pred_len - 1))
            mode_loss = []
            for k in range(y_true.shape[1]):  # number of traj
                mode_loss.append(self.space_loss(y_true[:, k], pred))
            mode_loss = tf.concat(mode_loss, axis=1)  # [B,K]
            mode_loss = tf.expand_dims(mode_loss, axis=-1)
            alphas.append(alpha)
            mode_losses.append(mode_loss)

        alphas = tf.concat(alphas, axis=-1)  # [B,M]
        mode_losses = tf.concat(mode_losses, axis=-1)  # [B,K,M]
        max_idx = tf.argmin(mode_losses, axis=-1)  # [B,K]
        epsilon = 0.05
        loss_matrix = tf.zeros_like(alphas)
        for k in range(y_true.shape[1]):
            selection_matrix = tf.one_hot(max_idx[:, k], depth=self.modes)
            loss_matrix = loss_matrix + (selection_matrix * mode_losses[:, k, :] * (1-epsilon))

        # considering all selected modes over all possible gt trajectories
        final_selection_matrix = tf.cast(tf.greater(loss_matrix, 0.0), tf.float32)  # [B,M]
        # give a cost to all trajectories which received no vote
        if self.modes > 1:
            relaxed_cost_matrix = (1. - final_selection_matrix) * mode_losses[:, 0, :] * epsilon / (float(self.modes) - 1.)
            final_cost_matrix = loss_matrix + relaxed_cost_matrix
        else:
            final_cost_matrix = loss_matrix
        trajectory_loss = tf.reduce_mean(tf.reduce_mean(final_cost_matrix, axis=-1))

        return trajectory_loss

class DiscretePositionLoss(Loss):
    def __init__(self):
        super(DiscretePositionLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        """
        y is a 3*out_seq_len dimensional vector with the position of future states
        The loss is the MSE of those two.
        """
        # Compute normalization factor (square lenght of the gt traj)
        average_loss = self.mse_loss(y_true, y_pred)
        average_loss = tf.expand_dims(average_loss, axis=-1)
        # Compute the batch averge loss. Multiplication by 10 seem to help optimization
        final_loss = average_loss * 10
        return final_loss

def convert_to_npy_traj(traj_csv_fname, num_states=10):
    npy_file = traj_csv_fname[:-4] + ".npy"
    if os.path.isfile(npy_file):
        # os.remove(npy_file)
        return npy_file
    df = pd.read_csv(traj_csv_fname, delimiter=',')
    if df.shape[0] == 0:
        return "None"

    x_pos_name = "pos_x_{}"
    y_pos_name = "pos_y_{}"
    z_pos_name = "pos_z_{}"

    x_pos_load = []
    y_pos_load = []
    z_pos_load = []
    for n in np.arange(1, num_states + 1):
        x_pos_load.append(x_pos_name.format(n))
        y_pos_load.append(y_pos_name.format(n))
        z_pos_load.append(z_pos_name.format(n))

    # Load data
    rel_cost = df["rel_cost"].values
    x_pos = df[x_pos_load].values
    y_pos = df[y_pos_load].values
    z_pos = df[z_pos_load].values

    # Each row is a different trajectory
    full_trajectory = np.column_stack((x_pos, y_pos, z_pos, rel_cost))
    # Write npy down
    assert full_trajectory.shape[-1] == 3 * num_states + 1
    np.save(npy_file, full_trajectory)
    return npy_file

def _convert_to_traj_sample(net_prediction, T_W_C, twist, config, network=True):
    net_prediction = np.array(net_prediction, dtype=float)
    # Conversion matrix to body frame
    R_W_C = T_W_C.R
    R_C_W = R_W_C.T
    v_W = np.array(twist).reshape((3, 1))
    v_C = R_C_W @ v_W
    bf_traj = []
    wf_traj = []

    net_prediction = net_prediction.reshape((config.state_dim, config.out_seq_len))
    # Initial point is just quad position
    point = TrajectoryPoint()
    point.heading = 0.0
    if config.ref_frame == 'wf':
        # a bit hacky, but works :)
        point.pose.position.x = T_W_C.t[0]
        point.pose.position.y = T_W_C.t[1]
        point.pose.position.z = T_W_C.t[2]
    elif config.ref_frame == 'bf':
        point.pose.position.x = 0.0
        point.pose.position.y = 0.0
        point.pose.position.z = 0.0

    point.pose.orientation.w = 1.0
    point.pose.orientation.x = 0.0
    point.pose.orientation.y = 0.0
    point.pose.orientation.z = 0.0

    bf_pose = fromPoseMessage(point)
    bf_traj.append(bf_pose)
    if config.ref_frame == 'wf':
        wf_pose = bf_pose
    elif config.ref_frame == 'bf':
        wf_pose = T_W_C * bf_pose
    wf_traj.append(wf_pose)
    # Add future points
    for k in range(config.out_seq_len):
        point = TrajectoryPoint()
        point.heading = 0.0
        point.pose.position.x = net_prediction[0, k]
        point.pose.position.y = net_prediction[1, k]
        point.pose.position.z = net_prediction[2, k]
        point.pose.orientation.w = 1.0
        point.pose.orientation.x = 0.0
        point.pose.orientation.y = 0.0
        point.pose.orientation.z = 0.0
        bf_pose = fromPoseMessage(point)
        bf_traj.append(bf_pose)
        if config.ref_frame == 'wf':
            wf_pose = bf_pose
        elif config.ref_frame == 'bf':
            wf_pose = T_W_C * bf_pose
        wf_traj.append(wf_pose)
    return bf_traj, wf_traj

def convert_to_trajectory(predictions, state, config, network=True):
    '''
    Predictions is a matrix of size [B, 10 * state_dim].
    This function converts them in a trajectory in (body and world frame).
    '''
    trajectories_list = []
    for b in range(predictions.shape[0]):
        sample = predictions[b]
        T_W_C_t = state[b][:3].reshape((3, 1))
        T_W_C_r = state[b][3:12].reshape((3, 3))
        T_W_C = Pose(T_W_C_r, T_W_C_t)
        twist = state[b][12:15]
        c_trajs = []
        for j in range(predictions.shape[1]):
            if network:
                alpha = sample[j, 0]
                traj_prediction = sample[j, 1:]
            else:
                alpha = 0.0
                traj_prediction = sample[j]
            traj_bf, traj_wf = _convert_to_traj_sample(traj_prediction, T_W_C, twist, config, network)
            c_trajs.append((traj_bf, traj_wf, alpha))
        trajectories_list.append(c_trajs)
    return trajectories_list

def transformToWorldFrame(trajectory, start_pos, start_att, ref_frame='bf'):
    if ref_frame == 'bf':
        T_W_S = Pose(start_att, start_pos)
    else:
        assert False, "Unknown reference frame."

    for i in range(trajectory.shape[1]):
        bf_pose = Pose(np.eye(3), trajectory[:, i].reshape((3, 1)))
        wf_pose = T_W_S * bf_pose
        trajectory[:, i] = np.squeeze(wf_pose.t)
    return trajectory

def save_trajectories(folder, trajectories, sample_num):
    # make folder
    write_folder = os.path.join(folder, "trajectories")
    if not os.path.isdir(write_folder):
        os.makedirs(write_folder)
    # Save header
    header = ""
    for j in range(len(trajectories[0][0][0])):
        header += 'pos_x_{},'.format(j)
        header += 'pos_y_{},'.format(j)
        header += 'pos_z_{},'.format(j)
    header += 'rel_cost'  # alpha in case of network

    for b in range(len(trajectories)):
        c_trajs = trajectories[b]  # c_trajs contains a set of (bf, wf, alpha) for each mode
        bf_fname = os.path.join(write_folder, "trajectories_bf_{:08d}.csv".format(sample_num[b]))
        wf_fname = os.path.join(write_folder, "trajectories_wf_{:08d}.csv".format(sample_num[b]))
        all_wf = []
        all_bf = []
        for k in range(len(c_trajs)):
            bf_traj = c_trajs[k][0]
            wf_traj = c_trajs[k][1]
            alpha = c_trajs[k][-1]
            bf_traj_np = []
            wf_traj_np = []
            for k in range(len(bf_traj)):
                bf_traj_np.append(bf_traj[k].t)
                wf_traj_np.append(wf_traj[k].t)
            bf_traj_np = np.concatenate(bf_traj_np).reshape(1, -1)
            wf_traj_np = np.concatenate(wf_traj_np).reshape(1, -1)
            # Alpha as cost (maybe exponent)
            cost = alpha
            bf_traj_np = np.append(bf_traj_np, cost * np.ones((1, 1)), axis=1)
            wf_traj_np = np.append(wf_traj_np, cost * np.ones((1, 1)), axis=1)
            all_bf.append(bf_traj_np)
            all_wf.append(wf_traj_np)
        all_bf = np.concatenate(all_bf)
        all_bf = all_bf[all_bf[:, -1].argsort()]
        all_wf = np.concatenate(all_wf)
        all_wf = all_wf[all_wf[:, -1].argsort()]
        np.savetxt(bf_fname, all_bf, fmt="%.8f", delimiter=',', header=header, comments='')
        np.savetxt(wf_fname, all_wf, fmt="%.8f", delimiter=',', header=header, comments='')
