class VehiclesDataset:
    def __init__(self):
        self.num_vehicle = 0
        self.num_object = 0
        self.num_object_with_kp = 0
        self.vehicles = dict()
        self.valid_ids = set()
        self.mean_shape = None
        self.pca_comp = None
        self.camera_mtx = None
        self.image_names = None
        self.data_dir = None
        self.mean_traj = None
        self.cov_traj = None
        self.plane = None

    def __str__(self):
        return "Vehicle Dataset: {} vehicles, {} objects".format(self.num_vehicle, self.num_of_objects())

    def insert_vehicle(self, id, vehicle):
        self.vehicles[id] = vehicle
        self.valid_ids.add(id)
        sorted(self.valid_ids)
        self.num_vehicle += 1

    def get_vehicle(self, query_id):
        if query_id not in self.valid_ids:
            return None
        else:
            return self.vehicles[query_id]

    def size(self):
        return self.num_vehicle

    def contains(self, query_id):
        return query_id in self.valid_ids

    def num_of_objects(self):
        num = 0
        for k, v in self.vehicles.items():
            num += v.num_objects
        self.num_object = num
        return num

    def num_of_objects_with_kp(self):
        num = 0
        for k, v in self.vehicles.items():
            num += v.num_objects_with_kp
        self.num_object_with_kp = num
        return num


class Vehicle:
    def __init__(self, image_path, keypoint, bbox, image_id, keypoint_pool):
        self.num_objects = 0
        self.num_objects_with_kp = 0
        self.id = None
        self.frames = dict()
        self.image_paths = dict()
        self.keypoints = dict()
        self.keypoints_backup = dict()
        self.keypoints_det2 = dict()
        self.keypoints_proj2 = dict()
        self.bboxs = dict()
        self.image_ids = dict()
        self.rt = dict()
        self.keypoints_pool = dict()
        self.insert_object(image_path, keypoint, bbox, image_id, keypoint_pool)
        self.pca = [0.0] * 5
        self.shape = [[0.0, 0.0, 0.0] * 12]
        self.spline = None  # np.zeros((6, ))
        self.spline_points = None
        self.spline_predict = None  # np.zeros((6, ))
        self.spline_points_predict = None
        self.rt_traj = dict()
        self.rotation_world2cam = None
        self.translation_world2cam = None
        self.first_appearance_frame_id = None
        self.stop_frame_range = None
        self.first_move_frame_id = None
        self.first_appearance_frame_time_pred = None
        self.traj_cluster_id = None

    def __str__(self):
        return "ID: {}, with {} objects".format(self.id, self.num_objects) + ', PCA: [' + \
               ', '.join(["{0:0.2f}".format(i) for i in self.pca]) + ']'

    def insert_object(self, image_path, keypoint, bbox, image_id, keypoint_pool=None, backup=False):
        if image_path in self.image_paths:
            print('{} is already contained, discard!'.format(image_path))
            return None
        else:
            object_id = self.num_objects
            self.image_paths[object_id] = image_path
            self.frames[object_id] = int(image_path[-8:-4])
            self.image_ids[object_id] = image_id
            if backup:
                self.keypoints_backup[object_id] = keypoint
            else:
                self.keypoints_backup[object_id] = None
            self.keypoints[object_id] = keypoint
            self.bboxs[object_id] = bbox
            self.rt[object_id] = None
            self.keypoints_pool[object_id] = keypoint_pool
            self.num_objects += 1
            if keypoint is not None:
                self.num_objects_with_kp += 1
            return object_id

    def set_id(self, init_id):
        self.id = init_id

    def set_pca(self, pca):
        if type(pca) is not list or len(pca) is not 5:
            raise Warning("PCA component should be list of length 5")
        else:
            self.pca = pca

    def set_3d_shape(self, shape):
        if type(shape) is not list or len(shape) is not 12:
            raise Warning("3D shape should be list of length 12, each has [x, y, z]")
        else:
            self.shape = shape

    def set_rt(self, obj_id, rvec, tvec):
        if type(rvec) is not list or len(rvec) is not 3 or type(tvec) is not list or len(tvec) is not 3:
            raise Warning("rvec and tvec should be list of length 3.")
        elif obj_id >= self.num_objects:
            raise Warning("object id doesnot exist.")
        else:
            self.rt[obj_id] = [rvec, tvec]

    def set_keypoints(self, obj_id, keypoints, backup=False):
        if len(keypoints) is not 12:
        # if type(keypoints) is not list or len(keypoints) is not 12:
            raise Warning("keypoints should be list of length 12.")
        elif obj_id >= self.num_objects:
            raise Warning("object id doesnot exist.")
        else:
            if backup:
                self.keypoints_backup[obj_id] = self.keypoints[obj_id]
            self.keypoints[obj_id] = keypoints


    def set_keypoints_cam2(self, obj_id, keypoints, det=True):
        if len(keypoints) is not 12:
            raise Warning("keypoints should be list of length 12.")
        elif obj_id >= self.num_objects:
            raise Warning("object id doesnot exist.")
        else:
            if det:
                self.keypoints_det2[obj_id] = keypoints
            else:
                self.keypoints_proj2[obj_id] = keypoints
