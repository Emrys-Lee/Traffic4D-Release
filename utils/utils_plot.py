import os
import sys
import numpy as np
import cv2

from dataset.vehicle import *
import config.params as params
from utils.utils_geom import world_to_cam, generate_spline_points_world, generate_spline_points_cam

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d
import random
from tqdm import tqdm

tableau10_extended_rgb = [(31, 119, 180), (255, 127, 14),
            (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75),
            (227, 119, 194), (127, 127, 127),
            (188, 189, 34),  (23, 190, 207)] + \
            [(0, 0, 255), (191, 255, 0), (255,255,0), (165,42,42)]
tableau10_extended = list(mcolors.TABLEAU_COLORS) + ['b', 'lime', 'yellow', 'brown']

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Arrow2D(FancyArrowPatch):
    def __init__(self, xs, ys, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts2d = xs, ys

    def draw(self, renderer):
        xs, ys = self._verts2d
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_car_proj_all(vehicle_dataset, params, suffix=''):
    if params.draw_keypoints_2d_num_frames <= 0:
        return

    image_data_dir = os.path.join(params.data_dir, "images")
    result_dir = os.path.join(params.result_dir, params.dataset_name+suffix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    valid_image_names = [n[0:5] for n in sorted(os.listdir(image_data_dir))][0:params.draw_keypoints_2d_num_frames]
    print("[draw_car_proj_all] Drawing vehicle 2D keypoints from " + image_data_dir)
    for i, image_name in tqdm(enumerate(valid_image_names)):
        image_path = os.path.join(image_data_dir, image_name + '.jpg')
        img = cv2.imread(image_path)
        output_img_name = os.path.join(result_dir, str(image_name) + '.jpg')

        # find which vehicles appear in the current frame
        for vehicle_id in vehicle_dataset.valid_ids:
            vehicle = vehicle_dataset.get_vehicle(vehicle_id)
            for object_id in range(vehicle.num_objects):
                if vehicle.image_paths[object_id][-9:] == image_path[-9:]:
                    box = vehicle.bboxs[object_id]
                    color_tableau = (0, 255, 0)
                    img = cv2.putText(img, 'ID:{}'.format(vehicle_id),
                                      (int(box[0]) + 20, int(box[1]) + 20),
                                      cv2.FONT_HERSHEY_COMPLEX, 1, color_tableau, 2, cv2.LINE_AA)
                    keypoints = vehicle.keypoints[object_id]
                    if keypoints is not None:
                        keypoints = np.array(keypoints)
                        draw_kp_12(img, keypoints, 0)
                    if box is not None:
                        draw_box(img, box)
        cv2.imwrite(output_img_name, img)
    print("[draw_car_proj_all] Vehicle 2D keypoints saved under " + result_dir)


def draw_traj_proj_all(vehicle_dataset, params, suffix=''):
    """trajectory projection, support both cam and sate view"""
    dataset_name = params.dataset_name
    affine_w2m = params.affine_w2m
    # load top view
    img = cv2.imread(params.data_dir + "/top_view.png")

    spline_points = []
    cluster_ids = []
    for veh_id, vehicle in vehicle_dataset.vehicles.items():
        if vehicle.spline is None:
            continue

        t_max = -1
        for obj_id, rt in vehicle.rt.items():
            if rt is not None:
                t = vehicle.image_ids[obj_id] - vehicle.first_appearance_frame_id
                t_max = max(t, t_max)
        spline_points.append(generate_spline_points_world(vehicle.spline, t_max, extend=False))
        cluster_ids.append(vehicle.traj_cluster_id)

    spline_points = np.array(spline_points)
    cluster_ids = np.array(cluster_ids)

    # print each cluster sequentially
    spline_points_temp = []
    cluster_ids_temp = []
    original_index = []
    for i in list(range(max(cluster_ids)+1)):
        index = np.where(cluster_ids==i)[0]
        if len(index) == 0:
            continue
        original_index += index.tolist()
        spline_points_temp.append(spline_points[index])
        cluster_ids_temp.append(cluster_ids[index])
    spline_points = np.concatenate(spline_points_temp, axis=0)
    cluster_ids = np.concatenate(cluster_ids_temp, axis=0)

    for i, (spline, cluster_id) in enumerate(zip(spline_points, cluster_ids)):
        spline_ = np.concatenate([spline[:, 0:2], np.ones((len(spline), 1))], axis=1)
        spline = (affine_w2m @ (spline_.T)).T
        if cluster_id < 0:
            continue
        for i in range(len(spline)-1):
            img = cv2.line(img, tuple(spline[i,0:2].astype(np.int32)), tuple(spline[i+1,0:2].astype(np.int32)),
                tableau10_extended_rgb[cluster_id][::-1], 2)

        i = len(spline) - 2
        img = cv2.arrowedLine(img, tuple(spline[i,0:2].astype(np.int32)), tuple(spline[i+1,0:2].astype(np.int32)),
            tableau10_extended_rgb[cluster_id][::-1], 2, tipLength = 1)

    # save output
    if not os.path.isdir(params.result_dir):
        os.makedirs(params.result_dir)
    out_img_name = dataset_name+'_top_view.jpg'
    cv2.imwrite(os.path.join(params.result_dir, out_img_name), img)
    print('[draw_traj_proj_all] Top view clusters saved as ' + os.path.join(params.result_dir, out_img_name))


def draw_box(img, box,color=(0,255,0)):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=1)


def draw_kp_12(img, keypoints, colorset=0, visibles=[1]*12):
    # draw connections
    con_3d = [[0, 2], [1, 3], [0, 1], [2, 3],
              [8, 10], [9, 11], [8, 9], [10, 11], [4, 5],
              [0, 4], [8, 4], [1, 5], [9, 5],
              [2, 6], [10, 6], [3, 7], [11, 7], [6, 7]]
    con_colors = []
    occ_color = (0, 0, 0)
    if colorset == 0:
        for _ in range(4):
            con_colors.append(tuple((255, 0, 0)))
        for _ in range(5):
            con_colors.append(tuple((0, 255, 0)))
        for _ in range(4):
            con_colors.append(tuple((0, 0, 255)))
        for _ in range(5):
            con_colors.append(tuple((255, 0, 255)))

    for i in range(len(con_3d)):
        i0, i1 = con_3d[i]
        if keypoints[i0,0] >= 0 and keypoints[i0,1] >= 0 and keypoints[i1,0] >= 0 and keypoints[i1,1] >= 0:
            cv2.line(img, tuple(keypoints[i0,0:2]), tuple(keypoints[i1,0:2]), con_colors[i], 2)

    # draw keypoints and text
    if colorset == 0:
        font_color = (0, 255, 0)
        kp_colors = [(128, 255, 0), (0, 255, 0), (64, 255, 255), (128, 255, 128),
                     (0, 255, 0), (128, 0, 0), (255, 0, 0), (255, 0, 128),
                     (255, 128, 128), (128, 128, 128), (0, 128, 255), (0, 255, 255)]

    for i in range(12):
        kp_color = occ_color if visibles[i]==0 else kp_colors[i]
        cv2.circle(img, tuple(keypoints[i, 0:2]), 3, kp_color, 3)
        cv2.putText(img, '{}'.format(i), tuple(keypoints[i, 0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)