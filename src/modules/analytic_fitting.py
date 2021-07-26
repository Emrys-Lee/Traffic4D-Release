import numpy as np
import cv2
from utils.utils_geom import r_to_unit_grad, cam_to_world_rt, generate_spline_points_cam, construct_mtx
from src.ceres.ceres_spline import *

def init_rt_each_veh(veh, params):
    rvecs = []
    tvecs = []
    frames = []
    for obj_id, rt in veh.rt.items():
        if rt is not None:
            rt = np.array(rt)
            rvec, tvec = rt[0, :], rt[1, :]
            rvecs.append(rvec)
            tvecs.append(tvec)
            frame = veh.image_ids[obj_id] - veh.first_appearance_frame_id
            frames.append(frame)

    if len(rvecs) < params.fitting_objects_thres:
        # too few frames are captured. Can't be used for analytic model fitting
        return None, None, None
    else:
        rvecs, tvecs, frames = np.array(rvecs), np.array(tvecs), np.array(frames)
        # print(rvecs, tvecs)
        return rvecs, tvecs, frames


def judge_valid_spline(spline_points, analytic_fitting_error, params):
    """Judge whether this spline is valid based on various conditions"""

    # condition 1: spline should be long enough
    long_enough = np.linalg.norm(spline_points[-1]-spline_points[0]) > params.fitting_length_thres

    # condition 2: vehicle's driving direction shouldn't change too much from its inital direction
    v = (spline_points[1:]-spline_points[:-1])
    v = v/np.linalg.norm(v, axis=1, keepdims=True) # [n,3]/[n]
    no_turn = True
    for i in range(1, v.shape[0]):
        if np.dot(v[0], v[i]) < -0.5:
            no_turn = False
            break

    # coondtion 3: spline fitting error should be small enough
    residual_enough = analytic_fitting_error < params.fitting_residual_thres

    cond =  long_enough and no_turn and residual_enough
    return cond


def solve_analytic_fitting(vd, params):
    print("[solve_analytic_fitting] Fitting analytic models to vehicles")
    for veh_id, veh in vd.vehicles.items():
        # clean spline field
        veh.spline = None
        veh.spline_points = None

        # load r,t and frame index from vd
        rvecs_cam, tvecs_cam, frames = init_rt_each_veh(veh, params)
        if frames is None:
            # too few frames are captured. Can't be used for analytic model fitting
            continue
        rvecs_world, tvecs_world = cam_to_world_rt(rvecs_cam, tvecs_cam, vd.rotation_world2cam, vd.translation_world2cam)
        # convert rotation vector into normalized gradient at each position
        unit_grad_world = r_to_unit_grad(rvecs_world)
        vts = np.concatenate([unit_grad_world, tvecs_world], axis=1)

        # some params for optimization
        spline = np.random.rand(params.fitting_num_spline_para)
        num_obj = len(rvecs_world)
        cost = np.zeros([1])
        # print("vts: {}".format(vts[0:20]))
        [spline, cost] = spline_fitting(vts.astype(np.float64),
                                  frames.astype(np.float64),
                                  spline.astype(np.float64),
                                  cost.astype(np.float64),
                                  num_obj,
                                  params.optimize_steps,
                                  params.fitting_coeff_reg
                                  )
        spline_points = generate_spline_points_cam(spline, vd.rotation_world2cam, vd.translation_world2cam, max(frames), extend=False)

        if judge_valid_spline(spline_points, cost[0]/num_obj, params):
            veh.spline = spline
            veh.spline_points = spline_points