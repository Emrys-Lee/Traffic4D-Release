import pickle
import numpy as np
import cv2
import os
from sklearn.neighbors import NearestNeighbors

from utils.utils_geom import construct_mtx, IOU, find_rt_c2w
from dataset.transform import update_state_vec_to_vehicle_dataset, convert_shape_to_veh3d
from src.ceres.ceres_reconstruct import *

def init_rt_pnp_ransac(vd, do_merge_det_pnp, mean_points_veh3d, init_mtx):
    """Generate initial state vector [r0 t0 r1 t1 ...] for vehicles
    using pnp ransac"""
    x0_rt = []
    # indices = []
    points_img_list = []           # 24*n, 2d keypoints detected
    points_proj_list = []          # 24*n, 2d keypoints projected by pnp
    inliers_mask_list = []         # 24*n, binary index for good keypoints
    valids_mask_list = []          # 24*n, binary index for keypoints inside frame
    index_to_veh_obj_id_list = []  # tuple(veh_id, obj_id)*n, map index to veh_id and obj_id
    index = 0

    print("[solve_longitudinal_reconstruction] Samples before optimization:")
    for veh_id in vd.valid_ids:
        vehicle = vd.get_vehicle(veh_id)
        for obj_id in range(vehicle.num_objects):
            bbox = vehicle.bboxs[obj_id]
            frame_id = vehicle.frames[obj_id]
            points_img = vehicle.keypoints[obj_id]
            if points_img is None:
                continue
            valid_points_index = np.where(np.logical_and(points_img[:,0] >= 0.0, points_img[:,1] >= 0.0))[0]
            if len(valid_points_index) < 4:
                continue
            points_img_valid = points_img[valid_points_index].astype(np.float32)
            mean_points_veh3d_valid = mean_points_veh3d[valid_points_index].astype(np.float64)
            # print(mean_points_veh3d_valid.dtype, mean_points_veh3d.dtype)

            # if points_2d is valid, run pnp ransac to get init r and t
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(mean_points_veh3d_valid,
                                                         points_img_valid,
                                                         init_mtx,
                                                         None,
                                                         reprojectionError=10.0,
                                                         flags=cv2.SOLVEPNP_EPNP)
            if inliers is None or len(inliers.squeeze()) <= -1: # 4:
                continue
            inliers = valid_points_index[inliers.squeeze()]
            inliers_mask = np.zeros((12,))
            inliers_mask[inliers.squeeze()] = 1 # inliers generated by pnp
            inliers_mask = inliers_mask.reshape((12,1))
            valid_mask = np.zeros((12,)) # originally keypoints
            valid_mask[valid_points_index] = 1
            valid_mask = valid_mask.reshape((12,1))

            points_proj, _ = cv2.projectPoints(mean_points_veh3d, rvec, tvec, init_mtx, None)
            points_proj = np.reshape(points_proj, (12,2))
            mean_error = np.linalg.norm(inliers_mask*(points_proj - points_img), ord=2) / np.sum(inliers_mask)
            points_proj_range = list(np.min(points_proj, axis=0)) + list(np.max(points_proj, axis=0))

            if mean_error < 300 and IOU(points_proj_range, bbox) > 0.3:#np.linalg.norm(tvec) < 1000:
                if index < 10:
                    with np.printoptions(precision=2, suppress=True):
                        print("Before - Index: {}, VehicleID: {}, ObjectID: {}, FrameID: {}, tvec: {}, rvec: {}, inliers: {}".format(index,
                            veh_id, obj_id, frame_id, tvec.squeeze(), rvec.squeeze(), inliers))
                x0_rt.extend(list(rvec.squeeze()))
                x0_rt.extend(list(tvec.squeeze()))
                inliers_mask_list.append(list(inliers_mask))
                valids_mask_list.append(list(valid_mask))
                points_proj_list.append(list(points_proj.squeeze()))
                points_img_list.append(list(points_img.squeeze()))
                index_to_veh_obj_id_list.append((veh_id, obj_id))

            index += 1

    x0_rt, points_img_list, points_proj_list, inliers_mask_list, valids_mask_list = \
            np.array(x0_rt), np.array(points_img_list), np.array(points_proj_list), np.array(inliers_mask_list), np.array(valids_mask_list)
    if do_merge_det_pnp:
        points_2d_list = points_proj_list * (1-inliers_mask_list) + points_img_list * inliers_mask_list
    else:
        points_2d_list = points_proj_list * (1-valids_mask_list) + points_img_list * valids_mask_list
    return x0_rt, points_img_list, points_proj_list, points_2d_list, inliers_mask_list, index_to_veh_obj_id_list


def init_pca_coeffs(index_to_veh_obj_id_list, params):
    index_to_pca_id_list = [] # n, index to pca id
    seen_veh_ids = set()
    pca_id = -1
    for (veh_id, obj_id) in index_to_veh_obj_id_list:
        if veh_id not in seen_veh_ids:
            seen_veh_ids.add(veh_id)
            pca_id += 1
        index_to_pca_id_list.append(pca_id)
    x0_pca = np.zeros(len(seen_veh_ids)*params.shape_num_pca_para)
    index_to_pca_id_list = np.array(index_to_pca_id_list, dtype=np.int32)
    return x0_pca, index_to_pca_id_list


def init_plane_and_Rt_c2w():
    plane = np.array([1.0, 1.0, 1.0, 1.0])
    Rt_c2w = None
    return plane, Rt_c2w


def solve_longitudinal_reconstruction(vd, params, init_only=False):
    dataset_name = params.dataset_name
    reprojection_scale_factor = params.optimize_reproj_scale
    plane_scale_factor = 10.0

    # Initalize camera matrix
    focal = params.focal
    height, width = params.size
    mtx = construct_mtx(focal, width, height)

    # Initialize state vectors
    mean_points_veh3d = convert_shape_to_veh3d(vd.mean_shape, params)
    x0_rt, points_img, points_proj, points_2d, inliers_mask, index_to_veh_obj_id= init_rt_pnp_ransac(vd,
           True,
           mean_points_veh3d,
           mtx)
    points_2d_reshape = points_2d.reshape(-1, 2)
    inliers_mask_reshape = inliers_mask.flatten()
     # print(inliers_mask_reshape)
    x0_pca, index_to_pca_id= init_pca_coeffs(index_to_veh_obj_id, params)
    plane, Rt_c2w = init_plane_and_Rt_c2w()

    # Initialize utils
    valid_num_obj = len(x0_rt) // 6
    valid_num_vehicle = len(x0_pca) // params.shape_num_pca_para

    if init_only:
        update_state_vec_to_vehicle_dataset(x0_rt, x0_pca, plane, vd, params, index_to_veh_obj_id, index_to_pca_id, mtx)
        return

    print("[solve_longitudinal_reconstruction] Start optimization\nrt: shape {}\npca: shape {}\nvalid_num_obj: {}\nvalid_num_vehicle: {}\npca_comp: shape {}\nmean_shape: shape {}\npoints_2d_reshape: shape {}\nindex_to_pca_id: shape {}\ninliers_mask: shape {}".format(
         x0_rt.shape,
         x0_pca.shape,
         valid_num_obj,
         valid_num_vehicle,
         vd.pca_comp.shape,
         vd.mean_shape.shape,
         points_2d_reshape.shape,
         index_to_pca_id.shape,
         inliers_mask_reshape.shape))

    # Global L-M optimization
    x_rt, x_pca, x_plane = optimize_pca_per_vehicle_new_pca(focal,
                                                            np.array(x0_rt).astype(np.float64),
                                                            np.array(x0_pca).astype(np.float64),
                                                            valid_num_obj,
                                                            valid_num_vehicle,
                                                            vd.pca_comp.astype(np.float64),
                                                            vd.mean_shape.astype(np.float64),
                                                            points_2d_reshape.astype(np.float64),
                                                            params.CAD_TO_METER,
                                                            np.array([height, width], dtype=np.float64),
                                                            np.array(plane, dtype=np.float64),
                                                            params.optimize_steps,
                                                            np.array(index_to_pca_id, dtype=np.int32),
                                                            np.array(inliers_mask_reshape, dtype=np.int32),
                                                            reprojection_scale_factor,
                                                            plane_scale_factor,
                                                            )
    # update rt, 3d shape, projected keypoints to vehicle dataset
    update_state_vec_to_vehicle_dataset(x_rt, x_pca, x_plane, vd, params, index_to_veh_obj_id, index_to_pca_id, mtx)