import cv2
import numpy as np
from dataset.vehicle import *
from utils.utils_geom import extract_Rt_world_to_cam


def convert_shape_to_veh3d(shape_3d, params):
    '''
    convert meanshape (13 dof) to [12, 3] represent 3D location of points
    :param shape_3d: array of size 13
    :return: ndarray [12, 3] represent 3D location of points
    '''
    shape_full = np.zeros((12, 3), dtype=np.float)
    base_front_length = shape_3d[0]
    base_back_length = shape_3d[1]
    base_half_width = shape_3d[2]
    base_height = shape_3d[3]
    top_front_length = shape_3d[4]
    top_back_length = shape_3d[5]
    top_half_width = shape_3d[6]
    top_height = shape_3d[7]
    mid_front_length = shape_3d[8]
    mid_back_length = shape_3d[9]
    mid_front_height = shape_3d[10]
    mid_back_height = shape_3d[11]
    mid_half_width = shape_3d[12]
    # base four keypoints
    shape_full[0, :] = np.array([base_front_length, base_height, -base_half_width])
    shape_full[1, :] = np.array([base_front_length, base_height, base_half_width])
    shape_full[2, :] = np.array([base_back_length, base_height, -base_half_width])
    shape_full[3, :] = np.array([base_back_length, base_height, base_half_width])

    # middle four keypoints
    shape_full[4, :] = np.array([mid_front_length, mid_front_height, -mid_half_width])
    shape_full[5, :] = np.array([mid_front_length, mid_front_height, mid_half_width])
    shape_full[6, :] = np.array([mid_back_length, mid_back_height, -mid_half_width])
    shape_full[7, :] = np.array([mid_back_length, mid_back_height, mid_half_width])

    # top four keypoints
    shape_full[8, :] = np.array([top_front_length, top_height, -top_half_width])
    shape_full[9, :] = np.array([top_front_length, top_height, top_half_width])
    shape_full[10, :] = np.array([top_back_length, top_height, -top_half_width])
    shape_full[11, :] = np.array([top_back_length, top_height, top_half_width])

    shape_full = shape_full * params.CAD_TO_METER
    return shape_full

def update_state_vec_to_vehicle_dataset(x_rt, x_pca, x_plane, vd, params, index_to_veh_obj_id, index_to_pca_id, mtx):
    # generate plane and world coordinate
    # world center is the average of all translation vectors. x,y axes lie on the ground, while z axis is vertical.
    center_world = np.average(x_rt.reshape((-1,6))[:,3:6], axis=0)
    R_w2c, t_w2c = extract_Rt_world_to_cam(x_plane, center_world)
    vd.rotation_world2cam = R_w2c
    vd.translation_world2cam = t_w2c
    vd.plane = x_plane
    vd.camera_mtx = mtx

    # print("x_rt: shape {}, context {}\nindex_to_veh_obj_id: shape {}".format(x_rt.shape, x_rt[0:20], len(index_to_veh_obj_id)))
    print("[solve_longitudinal_reconstruction] Samples after optimization:")
    for i, (veh_id, obj_id) in enumerate(index_to_veh_obj_id):
        # set rt
        rvec, tvec = x_rt[6*i:6*i+3], x_rt[6*i+3:6*i+6]
        veh = vd.get_vehicle(veh_id)
        veh.set_rt(obj_id, list(rvec), list(tvec))

        # set pca
        pca_id = index_to_pca_id[i]
        pca_coeff = x_pca[pca_id*params.shape_num_pca_para:pca_id*params.shape_num_pca_para+params.shape_num_pca_para]
        veh.set_pca(list(pca_coeff))
        shape_after_pca = vd.mean_shape + (pca_coeff[None, :] @ vd.pca_comp).squeeze()
        points_veh3d_after_pca = convert_shape_to_veh3d(shape_after_pca, params)
        # print(points_veh3d_after_pca)
        vd.get_vehicle(veh_id).set_3d_shape(list(points_veh3d_after_pca))

        # set keypoints
        points_proj, _ = cv2.projectPoints(points_veh3d_after_pca, rvec, tvec, mtx, None)
        # print(veh.keypoints[obj_id], points_proj)
        veh.set_keypoints(obj_id, points_proj.squeeze().astype(np.float32), backup=True)

        # set first appearance frame id
        frame = veh.image_ids[obj_id]
        first_appearance_frame_id = veh.first_appearance_frame_id
        veh.first_appearance_frame_id = frame if first_appearance_frame_id is None else min(frame, first_appearance_frame_id)

        # print results
        if i < 10:
            with np.printoptions(precision=2, suppress=True):
                print("After - Index: {}, VehicleID: {}, ObjectID: {}, tvec: {}, rvec: {}, PCA: {}".format(i,
                    veh_id, obj_id, tvec.squeeze(), rvec.squeeze(), pca_coeff))

