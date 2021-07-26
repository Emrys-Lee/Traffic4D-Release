import numpy as np
import cv2
import os
import pickle as pkl

def r_to_unit_grad(rvecs):
    unit_grad = []
    for r in rvecs:
        R = cv2.Rodrigues(r[np.newaxis,:])[0]
        theta_z = np.arctan2(R[1,0], R[0,0])
        unit_grad.append(np.array([np.cos(theta_z), np.sin(theta_z), 0]))
    unit_grad = np.array(unit_grad)
    return unit_grad


def extract_Rt_world_to_cam(plane, center):
    t = np.dot(plane[0:3], center) + plane[3]
    t = t / np.sum(np.square(plane[0:3]))
    center_proj = center - t * plane[0:3]
    t_world2cam = center_proj

    nz = -plane[0:3] / np.linalg.norm(plane[0:3])
    nx = np.array([1.0, 1.0, (-nz[0] - nz[1]) / (nz[2])])
    nx = nx / np.linalg.norm(nx)
    ny = np.cross(nz, nx)
    R = np.hstack((nx[:, None], ny[:, None], nz[:, None]))  # E
    R_world2cam = R
    return R_world2cam, t_world2cam


def world_to_cam(X_in_world, R_w2c, t_w2c):
    '''
    transform the world coordinate to the cam coordinate,
    world coordinate system is defined by the ground plane
    :param X_in_world: np.array of size (3,)
    :param R_w2c:
    :param t_w2c:
    :return:
    '''
    X_in_cam = R_w2c @ X_in_world + t_w2c
    return X_in_cam


def cam_to_world_rt(rvecs, tvecs, R_w2c, t_w2c):
    rt_car2cam = rt_vec_to_mat(rvecs, tvecs)
    # print("rt_car2cam", rt_car2cam)
    rt_world2cam = np.zeros([4, 4])
    rt_world2cam[0:3, 0:3] = R_w2c
    rt_world2cam[0:3, 3] = t_w2c
    rt_world2cam[3, 3] = 1.0
    # rt_vec_to_mat(rot_world2cam[np.newaxis,:], trans_world2cam[np.newaxis,:])
    # print("rt_world2cam", rt_world2cam)
    rt_car2world = np.linalg.inv(rt_world2cam) @ rt_car2cam
    rvecs, tvecs = rt_mat_to_vec(rt_car2world)

    # print(nx,ny,nz,R)
    # print("new center:", center_proj)
    # print("cam2world", np.linalg.inv(rt_world2cam)@np.array([1,1,26.15,1]))
    # exit(0)
    return rvecs, tvecs


def rt_seperate_to_mat(R, t):
    # R 3x3, return 4x4
    rt = np.concatenate([R, t.reshape((3, 1))], 1)
    rt = np.concatenate([rt, np.array([[0, 0, 0, 1]])], 0)
    return rt


def rt_vec_to_mat(rvecs, tvecs):
    single = False
    if (len(rvecs.shape) == 1):
        rvecs = rvecs[np.newaxis, :]
        single = True
    if (len(tvecs.shape) == 1):
        tvecs = tvecs[np.newaxis, :]
        single = True

    rt = np.zeros((len(tvecs), 4, 4))
    for n in range(len(tvecs)):
        (r, t) = (rvecs[n], tvecs[n])
        rt[n, 0:3, 0:3] = cv2.Rodrigues(r)[0]
        rt[n, 0:3, 3] = t
        rt[n, 3, 3] = 1.0
    if single:
        return rt[0]
    else:
        return rt


def rt_mat_to_vec(rt):
    single = False
    if (len(rt.shape) == 2):
        rt = rt[np.newaxis, :, :]
        single = True

    rvecs = []
    tvecs = []
    for n in range(len(rt)):
        R = rt[n, 0:3, 0:3]
        rvecs.append((cv2.Rodrigues(R)[0]).flatten())
        tvecs.append(rt[n, 0:3, 3])
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    if single:
        return rvecs[0], tvecs[0]
    else:
        return rvecs, tvecs


def find_rt_c2w(vd):
    R_w2c = vd.rotation_world2cam
    t_w2c = vd.translation_world2cam
    rt_w2c = rt_seperate_to_mat(R_w2c, t_w2c)
    rt_c2w = np.linalg.inv(rt_w2c)
    return rt_c2w


def generate_spline_points_cam(spline, R_w2c, t_w2c, end_frame, extend=True):
    if extend:
        t = np.arange(-5.0, end_frame + 5, 0.5)
    else:
        t = np.arange(0.0, end_frame, 0.5)
    t_power = np.stack([t ** 3, t ** 2, t, np.ones_like(t)]).T
    x = np.sum(spline[0:4] * t_power, axis=1)
    y = np.sum(spline[4:8] * t_power, axis=1)
    z = np.zeros_like(x)
    points_world = np.stack([x, y, z])  # 3*n
    # print("points in the world:\n", points_world.T)
    points_cam = (R_w2c @ points_world).T + t_w2c  # n*3
    # print("points in the cam:\n", points_cam)
    return points_cam


def generate_spline_points_world(splines, end_frame, extend=True, interval=0.5):
    if extend:
        t = np.arange(-5.0, end_frame + 5, interval)
    else:
        t = np.arange(0.0, end_frame, interval)
    single =  bool(len(splines.shape) == 1)
    if single:
        splines = splines[np.newaxis, :]
    points_world = []
    t_power = np.stack([t ** 3, t ** 2, t, np.ones_like(t)]).T
    for i in range(len(splines)):
        x = np.sum(splines[i,0:4] * t_power, axis=1, keepdims=True)
        y = np.sum(splines[i,4:8] * t_power, axis=1, keepdims=True)
        z = np.zeros_like(x)
        points_world.append(np.concatenate((x,y,z), axis=1))
    points_world = np.array(points_world)  # n_splines*3*n
    # print(points_world.shape)
    if single:
        points_world = points_world[0]
    return points_world


def generate_spline_points_given_frames(splines, t):
    single =  bool(len(splines.shape) == 1)
    if single:
        splines = splines[np.newaxis, :]
    n_spline_para = splines.shape[1]
    points_3d = []

    t_power = np.stack([t ** 3, t ** 2, t, np.ones_like(t)]).T
    for i in range(len(splines)):
        x = np.sum(splines[i,0:4] * t_power, axis=1, keepdims=True)
        y = np.sum(splines[i,4:8] * t_power, axis=1, keepdims=True)
        z = np.zeros_like(x)
        points_3d.append(np.concatenate((x,y,z), axis=1))

    points_3d = np.array(points_3d)  # n_splines*n*3
    # print(points_3d.shape)
    if single:
        points_3d = points_3d[0]
    return points_3d



def generate_spline_tangent_world(splines, end_frame, extend=True, interval=0.5):
    if extend:
        t = np.arange(-5.0, end_frame + 5, interval)
    else:
        t = np.arange(0.0, end_frame, interval)
    single =  bool(len(splines.shape) == 1)
    if single:
        splines = [splines]
    points_world = []
    t_power = np.stack([3*t ** 2, 2*t, np.ones_like(t), np.zeros_like(t)]).T
    for i in range(len(splines)):
        x = np.sum(splines[i,0:4] * t_power, axis=1, keepdims=True)
        y = np.sum(splines[i,4:8] * t_power, axis=1, keepdims=True)
        z = np.zeros_like(x)
        points_world.append(np.concatenate((x,y,z), axis=1))
    points_world = np.array(points_world)  # 3*n
    if single:
        points_world = points_world[0]
    return points_world


def construct_mtx(focal, width, height):
    mtx = np.array([[focal, 0, width/2],
                    [0, focal, height/2],
                    [0, 0, 1]])
    return mtx


def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou