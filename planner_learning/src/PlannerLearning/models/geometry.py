import numpy as np


def fixRotationMatrix(R):
    u, _, vt = np.linalg.svd(R)
    R_new = np.dot(u, vt)
    if np.linalg.det(R_new) < 0:
        R_new = -R_new
    return R_new


def geodesicDistanceSO3(R1, R2):
    return getRotationAngle(np.dot(R1, R2.T))


def getRotationAngle(R):
    return np.arccos((np.trace(R) - 1) / 2)
