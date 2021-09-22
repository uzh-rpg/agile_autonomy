import numpy as np
import pyquaternion
import scipy.linalg
#import geometry


def cross2Matrix(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def matrix2Cross(M):
    skew = (M - M.T) / 2
    return np.array([-skew[1, 2], skew[0, 2], -skew[0, 1]])


class Pose(object):
    def __init__(self, R, t):
        # Cannot be matrix or anything else, else chaos
        assert type(R) is np.ndarray
        assert type(t) is np.ndarray
        # For now, 2D could also be possible, but we need the following for
        # proper broadcasting in the transformation:
        assert t.shape[1] == 1
        self.R = R
        self.t = t

    def inverse(self):
        return Pose(self.R.T, -np.dot(self.R.T, self.t))

    def __mul__(self, other):
        # If the right operand is a pose, return the chained poses.
        if isinstance(other, Pose):
            return Pose(np.dot(self.R, other.R), np.dot(self.R, other.t) + self.t)
        # If it is a vector or several vectors expressed as matrix, apply the
        # pose transformation to them.
        if type(other) is np.ndarray or \
                type(other) is np.matrixlib.defmatrix.matrix:
            assert len(other.shape) == 2
            assert other.shape[0] == 3
            return np.dot(self.R, other) + self.t

        raise Exception('Multiplication with unknown type!')

    def asArray(self):
        return np.vstack((np.hstack((self.R, self.t)), np.array([0, 0, 0, 1])))

    def asTwist(self):
        so_matrix = scipy.linalg.logm(self.R)
        if np.sum(np.imag(so_matrix)) > 1e-10:
            raise Exception('logm called for a matrix with angle Pi. ' +
                            'Not defined! Consider using another representation!')
        so_matrix = np.real(so_matrix)
        return np.hstack((np.ravel(self.t), matrix2Cross(so_matrix)))

    def q_wxyz(self):
        return pyquaternion.Quaternion(matrix=self.R).unit.q

    #def fix(self):
    #    self.R = geometry.fixRotationMatrix(self.R)


def fromMatrix(M):
    return Pose(M[:3, :3], M[:3, 3].reshape(3, 1))


#def fromApproximateMatrix(M):
#    return Pose(geometry.fixRotationMatrix(M[:3, :3]), M[:3, 3].reshape(3, 1))


def fromTwist(twist):
    # Using Rodrigues' formula
    w = twist[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return Pose(np.eye(3), twist[:3].reshape(3, 1))
    M = cross2Matrix(w / theta)
    R = np.eye(3) + M * np.sin(theta) + np.dot(M, M) * (1 - np.cos(theta))
    return Pose(R, twist[:3].reshape(3, 1))


# ROS geometry_msgs/Pose
def fromPoseMessage(pose_msg):
    pos = pose_msg.pose.position
    ori = pose_msg.pose.orientation
    R = pyquaternion.Quaternion(ori.w, ori.x, ori.y, ori.z).rotation_matrix
    t = np.array([pos.x, pos.y, pos.z]).reshape(3, 1)
    return Pose(R, t)


# ROS geometry_msgs/PoseStamped, also returns stamp
def fromPoseStamped(pose_stamped):
    return fromPoseMessage(pose_stamped.pose), pose_stamped.header.stamp


# ROS geometry_msgs/Transform
def fromTransformMessage(tf_msg):
    pos = tf_msg.translation
    ori = tf_msg.rotation
    R = pyquaternion.Quaternion(ori.w, ori.x, ori.y, ori.z).rotation_matrix
    t = np.array([pos.x, pos.y, pos.z]).reshape(3, 1)
    return Pose(R, t)


# ROS tf2_msgs/TFMessage, also returns stamp
def fromTf(tf_msg, parent_frame, child_frame):
    """Currently assumes parent to child is explicitly in the tf message.
    
    For other cases, it might be better to run
    http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20listener%20%28Python%29
    as the intermediate transforms might not even all be in the same message.
    """
    for tform in tf_msg.transforms:
        if tform.header.frame_id == parent_frame and \
                tform.child_frame_id == child_frame:
            return fromTransformMessage(tform.transform), tform.header.stamp
    return None, None


def cosSinDeg(angle_deg):
    return np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))


def xRotationDeg(angle_deg):
    c, s = cosSinDeg(angle_deg)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    return Pose(R, np.zeros((3, 1)))


def yRotationDeg(angle_deg):
    c, s = cosSinDeg(angle_deg)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return Pose(R, np.zeros((3, 1)))


def zRotationDeg(angle_deg):
    c, s = cosSinDeg(angle_deg)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return Pose(R, np.zeros((3, 1)))


def rollPitchYawDeg(r, p, y):
    return xRotationDeg(r) * yRotationDeg(p) * zRotationDeg(y)


def yawPitchRollDeg(y, p, r):
    return zRotationDeg(y) * yRotationDeg(p) * xRotationDeg(r)


def identity():
    return Pose(np.eye(3), np.zeros((3, 1)))


def translation(t):
    return Pose(np.eye(3), np.array(t).reshape((3, 1)))
