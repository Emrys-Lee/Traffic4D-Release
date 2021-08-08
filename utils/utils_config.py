import yaml

class Params:
    def __init__(self):
        # image size
        self.size = (1080, 1920)
        # focal length
        self.focal = 2098.0

        ##### shape model
        # number of pca components in shape
        self.shape_num_pca_para = 5
        # CAD scale to meter scale
        self.CAD_TO_METER = 7.107843137
        # number of spline params. A 3-order spline has 8 params
        self.fitting_num_spline_para = 8
        # number of optimize steps for LM
        self.optimize_steps = 100

        self.optimize_reproj_scale = 2.0

        self.fitting_coeff_reg = 500.0

        self.fitting_length_thres = 5.0

        self.fitting_objects_thres = 6

        self.fitting_residual_thres = 100.0

        self.cluster_init_num_first = 6

        self.cluster_init_num_second = 6

        self.cluster_group_size_thres = 10

        self.cluster_proj_diff_thres = 3.0

        self.cluster_projection_t = [0.0, 10.0]

        self.data_dir = ""

        self.result_dir = ""

        self.dataset_name = ""

        self.affine_w2m = []

        self.draw_keypoints_2d_num_frames = 5

    def yml_load(self, yml_name):
        with open(yml_name, 'r') as f:
            yml_params = yaml.load(f)
        for k, v in yml_params.items():
            if (k == "focal"):
                self.focal = v
            elif (k == "optimize_reproj_scale"):
                self.optimize_reproj_scale = v
            elif (k == "fitting_coeff_reg"):
                self.fitting_coeff_reg = v
            elif (k == "fitting_length_thres"):
                self.fitting_length_thres = v
            elif (k == "fitting_residual_thres"):
                self.fitting_residual_thres = v
            elif (k == "cluster_init_num_first"):
                self.cluster_init_num_first = v
            elif (k == "cluster_init_num_second"):
                self.cluster_init_num_second = v
            elif (k == "cluster_group_size_thres"):
                self.cluster_group_size_thres = v
            elif (k == "cluster_proj_diff_thres"):
                self.cluster_proj_diff_thres = v
            elif (k == "affine_w2m"):
                self.affine_w2m = v
            elif (k == "dataset_name"):
                self.dataset_name = v
            elif (k == "data_dir"):
                self.data_dir = v
            elif (k == "result_dir"):
                self.result_dir = v
            elif (k == "draw_keypoints_2d_num_frames"):
                self.draw_keypoints_2d_num_frames = v


