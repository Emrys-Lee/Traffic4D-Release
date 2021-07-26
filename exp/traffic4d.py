import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils_save import *
from utils.utils_plot import draw_car_proj_all, draw_traj_proj_all
from utils.utils_config import Params
from src.modules.longitudinal_reconstruction import solve_longitudinal_reconstruction
from src.modules.analytic_fitting import solve_analytic_fitting
from src.modules.activity_clustering import solve_longitudinal_clustering

def reconstruction_go(params):
    # load dataset
    vd = vd_load(params, suffix='_init', in_result=False)
    # init rt and do LM optimization
    solve_longitudinal_reconstruction(vd, params)
    vd_dump(vd, params, suffix='_rec')
    draw_car_proj_all(vd, params, suffix='_keypoints')

def analytic_fitting_go(params):
    vd = vd_load(params, suffix='_rec')
    solve_analytic_fitting(vd, params)
    vd_dump(vd, params, suffix='_analytic')

def activity_clustering_go(params):
    vd = vd_load(params, suffix='_rec')
    solve_analytic_fitting(vd, params)
    vd_dump(vd, params, suffix='_analytic')

    solve_longitudinal_clustering(vd, params)
    vd_dump(vd, params, suffix='_clusters')

    draw_traj_proj_all(vd, params)

if __name__ == "__main__":
    np.random.seed(seed=800)
    np.set_printoptions(precision=4, suppress=True)

    if (len(sys.argv) != 3):
        print("Usage: python exp/traffic4d.py [yml path] [action]")
        exit(1)

    yml_name = sys.argv[1]
    action = sys.argv[2]

    params = Params()
    if os.path.isfile(yml_name):
        params.yml_load(yml_name)
    else:
        print("[yml path] not exist")
        exit(1)

    if action == "reconstruction":
        reconstruction_go(params)
    elif action == "clustering":
        activity_clustering_go(params)
    else:
        print("[action] unknown")
        exit(1)
    print("Done")


