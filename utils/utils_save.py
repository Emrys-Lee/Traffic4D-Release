import pickle as pkl
import os

def pkl_dump(obj, name):
    with open(name, 'wb') as f:
        pkl.dump(obj, f)


def pkl_load(name):
    obj = None
    with open(name, 'rb') as f:
        obj = pkl.load(f)
    assert(obj is not None)
    return obj

def vd_dump(vd, params, suffix='', in_result=True):
    result_dir = params.result_dir if in_result else params.data_dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    vd_name = os.path.join(result_dir, params.dataset_name + suffix + '.vd')
    pkl_dump(vd, vd_name)
    print("[vd_dump] Vehicle dataset saved as {}".format(vd_name))

def vd_load(params, suffix='', in_result=True):
    vd_name = os.path.join(params.result_dir if in_result else params.data_dir, params.dataset_name + suffix + '.vd')
    print("[vd_load] Loading {}".format(vd_name))
    if os.path.isfile(vd_name):
        vd = pkl_load(vd_name)
    else:
        vd = None
    return vd
