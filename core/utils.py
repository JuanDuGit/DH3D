from __future__ import division, print_function

import math
import numpy as np
from tabulate import tabulate
import os
import json
from termcolor import colored
from tensorpack import logger
import pickle
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def single_nms(xyz, attention, nms_radius, min_response_ratio, max_keypoints, remove_noise=True):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    if remove_noise:
        dist_check = distances[:, 7]
        invalid = dist_check > 2.0
        attention[invalid] = 0.0

    knn_attention = attention[indices]
    outside_ball = distances > nms_radius
    knn_attention[outside_ball] = 0.0
    is_max = np.where(np.argmax(knn_attention, axis=1) == 0)[0]

    # Extract the top k features, filtering out weak responses
    attention_thresh = np.max(attention) * min_response_ratio

    is_max_attention = [(attention[m], m) for m in is_max if attention[m] > attention_thresh]
    is_max_attention = sorted(is_max_attention, reverse=True)

    max_indices = [m[1] for m in is_max_attention]

    if len(max_indices) >= max_keypoints:
        max_indices = max_indices[:max_keypoints]
        num_keypoints = len(max_indices)
    else:
        num_keypoints = len(max_indices)  # Retrain original number of points
    # print('num_keypoints:', num_keypoints)
    return num_keypoints, max_indices


def get_sets_dict(filename):
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        # print("number of item: {}.\n".format(len(trajectories.keys())))
        return trajectories


def get_knn(positions, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    return indices, distances


def mkdir_p(dirname):
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def log_config_info(config):
    config_keys = sorted(config.keys())
    data = []
    for k in config_keys:
        data.append([k, config[k]])
    headers = ['config_name', 'content']

    table = tabulate(data, headers=headers)
    logger.info(colored("List of Config Args: \n", 'cyan') + table)

    # save as json
    writefile = os.path.join(logger.get_logger_dir(), 'config.json')
    with open(writefile, 'w') as f:
        json.dump(config, f)


def get_fixednum_pcd(cloud, targetnum, randsample=True, need_downsample=False, sortby_dis=True):
    if need_downsample:
        cloud = downsample(cloud)
    ind = remove_noise(cloud)
    cloud = cloud[ind, :]
    ori_num = cloud.shape[0]
    if cloud.shape[0] > targetnum:
        if sortby_dis:
            centroid = np.mean(cloud, axis=0)
            dis = np.sum(np.square(cloud - centroid), axis=1)
            ind = np.argsort(dis)
            cloud = cloud[ind[0:targetnum], :3]
        choice_idx = np.random.choice(cloud.shape[0], targetnum, replace=False)
        cloud = cloud[choice_idx, :]
        ori_num = targetnum
    else:
        num_to_pad = targetnum - cloud.shape[0]
        if randsample:
            pad_points = cloud[np.random.choice(cloud.shape[0], size=num_to_pad, replace=True), :]
        else:
            pad_points = np.ones([num_to_pad, 3], dtype=np.float32) * 100000

        cloud = np.concatenate((cloud, pad_points), axis=0)
    return cloud, ori_num


def center_crop(pc, rad=20, shift=[0, 0, 0]):
    centroid = np.mean(pc, axis=0)
    centroid += shift
    mask = np.sum(np.square(pc - centroid), axis=1) <= rad * rad
    pc = pc[mask, :]
    return pc


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts_ind = []
        farthest_pts_ind.append(np.random.randint(len(pts)))
        first_point = pts[farthest_pts_ind[0]]
        distances = self.calc_distances(first_point, pts)
        for i in range(1, k):
            farthest_pts_ind.append(np.argmax(distances))
            distances = np.minimum(distances, self.calc_distances(pts[farthest_pts_ind[i]], pts))
        return np.asarray(farthest_pts_ind)


def load_descriptor_bin(filename, dim=131, dtype=np.float32):
    desc = np.fromfile(filename, dtype=dtype)
    desc = np.reshape(desc, (-1, dim))
    return desc


def load_single_pcfile(filename, dim=3, dtype=np.float32):
    pc = np.fromfile(filename, dtype=dtype)
    pc = np.reshape(pc, (pc.shape[0] // dim, dim))
    return pc[:, 0:3]


def write_to_bin(points, filename):
    with open(filename, 'wb') as f:
        points.tofile(f)


def restore_scale_pcd(pcd, knn=3):
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(pcd[:, 0:3])
    distances, indices = nbrs.kneighbors(pcd[:, :])  # ptnnum1, n_neighbors
    distances_sum = np.mean(distances)
    scale = 0.2 / distances_sum
    pcd *= scale
    return pcd


def downsample(pcd, voxelsize=0.2):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd)
    cloud_down = cloud.voxel_down_sample(voxel_size=voxelsize)
    cloud_down_p = np.asarray(cloud_down.points)
    return cloud_down_p


def remove_noise(pcd, nb_points=4, radius=1.0):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd)
    cl, ind = cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return ind


def plot_pc(s):
    if not isinstance(s, np.ndarray):
        s = np.asarray(s.points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(s)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [pcd, origin_frame])


def plot_pc_pair(s, t):
    if not isinstance(s, np.ndarray):
        s = np.asarray(s.points)
    if not isinstance(t, np.ndarray):
        t = np.asarray(t.points)

    red = np.asarray([255, 0, 0])
    blue = np.asarray([0, 0, 225])
    s_colors = np.tile(red, [s.shape[0], 1]) / 255
    t_colors = np.tile(blue, [t.shape[0], 1]) / 255
    points = np.vstack([s, t])
    colors = np.vstack([s_colors, t_colors])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [pcd, origin_frame])


def apply_transform(pcd, transform):
    if not isinstance(pcd, np.ndarray):
        pcd = np.asarray(pcd.points)
    pcd = pcd.transpose()  # to 3*n
    pcd = np.vstack([pcd, np.ones([1, pcd.shape[1]])])  # 4 * pointnum
    pcd = np.dot(transform, pcd)  # 4* n
    pcd = pcd.transpose()
    return pcd[:, 0:3]


# *************************************** angle err************************************#
def angle_error1(R1, R2):
    cos = (np.trace(np.dot(np.linalg.inv(R1), R2)) - 1) / 2
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error2(R1, R2):
    euler = euler_from_matrix(np.dot(np.linalg.inv(R1), R2))
    err_rot = np.sum(abs(euler))
    return err_rot


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        import IPython
        IPython.embed()

    return err_q, err_t

def rigid_transform_3D(A, B, return44=False):
    '''
     v0: n*3
     Rt    - The rotation matrix such that A = R * B + t
    '''

    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    R12 = BB - AA
    R21 = AA.T - BB.T
    R22_1 = BB.T + AA.T
    R22 = crossTimesmatrix(R22_1[0:3, :].T)

    B = np.zeros((4, 4))
    A = np.zeros((N, 4, 4))

    for i in range(N):
        A[i, 0, 1:4] = R12[i, 0:3]
        A[i, 1:4, 0] = R21[0:3, i]
        A[i, 1:4, 1:4] = R22[i, 0:3, 0:3]
        B = B + np.dot(A[i, :, :].T, A[i, :, :])

    U, S, Vt = np.linalg.svd(B)
    quat = Vt[3, :]
    rot = quaternion_matrix(quat)[0:3, 0:3]
    T = np.zeros((4, 4))
    T[0:3, 0:3] = rot
    T[0:3, 3] = centroid_A - np.dot(rot, centroid_B)
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]
    if return44:
        return T
    else:
        return R, t

# ***************************************start of transform.py************************************#
def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, np.pi/2.0)
    >>> np.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = np.random.random(3)
    >>> n = vector_norm(v)
    >>> np.allclose(n, np.linalg.norm(v))
    True
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> np.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(np.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2) * r2, np.sin(t1) * r1,
                     np.cos(t1) * r1, np.sin(t2) * r2])


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not np.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


# ***************************************end of transform.py************************************#

def crossTimesmatrix(v):
    a = v.shape[0]
    b = v.shape[1]
    v_times = np.zeros((a, 3, b))
    v_times[:, 0, 1] = -v[:, 2]
    v_times[:, 0, 2] = v[:, 1]
    v_times[:, 1, 0] = v[:, 2]
    v_times[:, 1, 2] = -v[:, 0]
    v_times[:, 2, 0] = - v[:, 1]
    v_times[:, 2, 1] = v[:, 0]
    return v_times



