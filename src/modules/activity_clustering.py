import numpy as np
import cv2
from sklearn import mixture
from utils.utils_geom import r_to_unit_grad, generate_spline_points_given_frames, generate_spline_tangent_world
from utils.utils_save import *

def merge_similar_clusters_first(means, index_global):
    """merge clusters with similiar trajectory directions"""
    if means.shape[1] == 8:
        # means in splines
        tangents = generate_spline_tangent_world(means, 10.0) # shape: [n_cluster, n_points, 3]
        tangents = np.average(tangents, axis=1)
    else:
        # means in spline points
        tangents = means[:,1:] - means[:,:-1]
        tangents = np.average(tangents, axis=1)
    angle_tangents = np.arctan2(tangents[:,1], tangents[:,0])/np.pi*180.0
    angle_tangents *= 1

    # merge clusters according to tangents
    n_components_root = len(means)
    n_components_root_merged = n_components_root
    alpha_tangent_thres = 10.0
    cluster_merged = list(range(n_components_root))
    for cluster in range(1, n_components_root):
        for prev_cluster in range(cluster):
            inlier_num_enough = np.sum(np.array(index_global==cluster, dtype=np.int32)) >= 4
            if cluster==1 and prev_cluster==0 and np.sum(np.array(index_global==prev_cluster, dtype=np.int32)) < 4:
                inlier_num_enough = False
            if abs(angle_tangents[cluster] - angle_tangents[prev_cluster]) < alpha_tangent_thres or (not inlier_num_enough):
                cluster_merged[cluster] = cluster_merged[prev_cluster]
                break
        cluster_merged[cluster] = min(cluster_merged[cluster],
                max(cluster_merged[0:cluster])+1)

    for cluster in range(n_components_root):
        index = np.where(index_global==cluster)[0]
        index_global[index] = cluster_merged[cluster]
    return index_global


def spline_projection(proj_vec, spline_points_world):
    """Project trajectories to 1D"""
    mx, my = proj_vec
    # spline_points # []
    spline_points_world2d = spline_points_world[:,:,0:2] #n_splines*(x,y)*(t1,t2)
    spline_points_proj1d = (spline_points_world2d[:,:,0]*mx + spline_points_world2d[:,:,1]*my) / np.linalg.norm(proj_vec)
    return spline_points_proj1d


def merge_similar_clusters_second(means, index_global, params):
    index_global = np.array(index_global)
    n_components_root = len(means)
    cluster_merged = list(range(n_components_root))
    for cluster in range(1, n_components_root):
        for prev_cluster in range(cluster):
            mean_diff_enough = np.linalg.norm(means[cluster]-means[prev_cluster]) > params.cluster_proj_diff_thres
            inlier_num_enough = np.sum(np.array(index_global==cluster, dtype=np.int32)) >= params.cluster_group_size_thres
            if cluster==1 and prev_cluster==0 and np.sum(np.array(index_global==prev_cluster, dtype=np.int32)) < params.cluster_group_size_thres:
                inlier_num_enough = False
            if (not mean_diff_enough) or (not inlier_num_enough):
                cluster_merged[cluster] = cluster_merged[prev_cluster]
                break
        cluster_merged[cluster] = min(cluster_merged[cluster],
                max(cluster_merged[0:cluster])+1)

    for cluster in range(n_components_root):
        index = np.where(index_global==cluster)[0]
        index_global[index] = cluster_merged[cluster]
    return index_global


def cluster_projection(splines, params, veh_ids, vd):
    """Project trajectories to 1D and cluster them"""
    cluster_ids_init = np.zeros([len(splines)])
    # one point: no need to cluster, return cluster id as 0
    if len(splines)==1:
         return cluster_ids_init

    vi = veh_ids[0]
    veh = vd.get_vehicle(vi)
    last_frame_id = veh.first_appearance_frame_id
    for obj_id, rt in veh.rt.items():
        if rt is not None:
            last_frame_id = veh.image_ids[obj_id]
    frame = last_frame_id - veh.first_appearance_frame_id
    t = np.array([0.0, frame])
    t = np.arange(params.cluster_projection_t[0], params.cluster_projection_t[1], 1.0)

    spline_points = generate_spline_points_given_frames(splines, t) # n_splines*2*3
    # print("spline points", spline_points)

    # project splines to 1d (similiar to Fisher LDA)
    average_v = np.average(spline_points[:,1,:] - spline_points[:,0,:], axis=0)
    average_v /= np.linalg.norm(average_v)
    average_v_list.append(average_v)
    proj_vec = np.array([-average_v[1], average_v[0]])
    splines_points_proj = spline_projection(proj_vec, spline_points)
    # print("spline points projected shape", splines_points_proj.shape) # n_splines, 2

    # clustering
    dpgmm = mixture.BayesianGaussianMixture(n_components=min(len(splines), params.cluster_init_num_second),
                                                covariance_type='diag').fit(splines_points_proj)
    means = dpgmm.means_

    # judge whether clustering is valid. If so, return new cluster ids, otherwise return all cluster ids as 0
    cluster_ids = dpgmm.predict(splines_points_proj)
    # valid_cluster = judge_valid_cluster_projection(means, cluster_ids, dataset_name)
    cluster_ids = merge_similar_clusters_second(means, cluster_ids, params)
    # return cluster_ids if valid_cluster else cluster_ids_init
    return cluster_ids


def cluster_first_level(splines, params):
    """first level cluster - split diretions. Perform GMM clustering on splines, and then merge clusters with similiar directions"""

    coeff_cluster_scale = np.repeat([1.0, 1.0, 1.0, 1.0], 2)
    splines_scaled = np.array(splines) * coeff_cluster_scale
    dpgmm = mixture.BayesianGaussianMixture(n_components=params.cluster_init_num_first,
                                                covariance_type='diag').fit(splines_scaled)
    cluster_ids = dpgmm.predict(splines_scaled)
    means = dpgmm.means_ / coeff_cluster_scale
    cluster_ids = merge_similar_clusters_first(means, cluster_ids)
    return cluster_ids


def cluster_second_level(cluster_ids, splines, vd, params):
    """second level cluster - split lanes in one direction. Perform Fisher projection on splines in each first level cluster, and then GMM cluster these projected splines"""
    n_components_first= max(cluster_ids) + 1
    index_list = [[] for i in range(n_components_first)]
    splines_list =  [[] for i in range(n_components_first)]
    veh_id_list = [[] for i in range(n_components_first)]
    global average_v_list
    average_v_list = []

    # split splines according to first level clusters
    for index, cluster in enumerate(cluster_ids):
        index_list[cluster].append(index)
        splines_list[cluster].append(splines[index])
        veh_id_list[cluster].append(index_to_veh_id[index])

    # perform second level clustering for each
    for cluster_id_first in range(n_components_first):
        splines_second = np.array(splines_list[cluster_id_first])
        cluster_ids_second = cluster_projection(splines_second, params, veh_id_list[cluster_id_first], vd)
        delta_cluster_id = max(cluster_ids)
        for index_second, cluster_id_second in enumerate(cluster_ids_second):
            index_first = index_list[cluster_id_first][index_second]
            if cluster_id_second!=0:
                cluster_ids[index_first] = delta_cluster_id + cluster_id_second

    return cluster_ids


def cluster_clean_up(splines, cluster_ids):
    """compute mixture model based on cluster_ids given by hierachical clustering"""
    n_clusters = max(cluster_ids) + 1
    means = []
    covariances = []
    global_probs = []

    for cluster in range(n_clusters):
        indices = np.where(cluster_ids==cluster)[0]

        # otherwise compute cluster
        splines_cluster = splines[indices]
        gaussian = mixture.GaussianMixture(n_components=1,
                covariance_type='diag').fit(splines_cluster)
        mean = gaussian.means_[0]
        covariance = gaussian.covariances_[0]

        means.append(mean)
        covariances.append(covariance)

    return means, covariances


def cluster_hierachical(splines, vd, params):
    # first level cluster - split diretions. Perform GMM clustering on splines, and then merge clusters with similiar directions
    cluster_ids = cluster_first_level(splines, params)
    print("[solve_longitudinal_clustering] First level cluster_ids:\n", cluster_ids)
    # second level cluster - split lanes in one direction. Perform Fisher projection on splines in each first level cluster, and then GMM cluster these projected splines
    cluster_ids = cluster_second_level(cluster_ids, splines, vd, params)
    # clean up: once we have hierachical clusters, for each cluster we calculate a Gaussian component from all its trajectories, so we have the final mixture model
    mean, covariance = cluster_clean_up(splines, cluster_ids)
    print("[solve_longitudinal_clustering] Second level cluster_ids:\n", cluster_ids)

    return cluster_ids, mean, covariance


def init_splines_to_cluster(vd):
    global index_to_veh_id
    index_to_veh_id = []
    splines = []
    for veh_id, veh in vd.vehicles.items():
        veh.traj_cluster_id = None
        if veh.spline is not None:
            index_to_veh_id.append(veh_id)
            splines.append(veh.spline)
    splines = np.array(splines)
    return splines, index_to_veh_id


def solve_longitudinal_clustering(vd, params):
    np.random.seed(seed=800)
    splines, index_to_veh_id = init_splines_to_cluster(vd)
    cluster_ids, mean, covariance = cluster_hierachical(splines, vd, params)
    vd.mean_traj, vd.cov_traj = mean, covariance
    for index, cluster_id in enumerate(cluster_ids):
        veh_id = index_to_veh_id[index]
        vd.get_vehicle(veh_id).traj_cluster_id = cluster_id

