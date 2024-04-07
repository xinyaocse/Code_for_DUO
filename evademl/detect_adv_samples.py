import numpy as np
from sklearn.neighbors import KernelDensity
from evademl.detect_adv_utils import *
import glob
from network.symbol_builder import dataset_cfg
import os

def read_path(save_path, fpath_label):
    f = open(fpath_label)
    l = f.read().splitlines()
    f.close()
    # print l
    # 对fpath与labels进行处理
    fpaths = list()
    labels = list()

    for count, item in enumerate(l):
        v_id, label, path = item.split()
        fpaths.append(os.path.join(save_path, path.split('/')[1].split('.')[0]))
        labels.append(int(label))

        if count==500:
            break

    return fpaths,labels


def get_data_path(file_path,type='_ori.npy'):
    new_file_path = []
    for file in file_path:
        new_file_path.append(file + type)
    return new_file_path

# 输入的是 numpy
def load_data(file_path):
    data = np.load(file_path[0])
    for idx in range(1, len(file_path)):
        temp = np.load(file_path[idx])
        data = np.concatenate([data,temp], axis=0)

    return data


def detect_adv(X1,model,parser):
    root_folder, train_fpath_label, test_fpath_label, db_fpath_label, classes \
                 = dataset_cfg(parser)
    save_path = f'./advdataset/{parser.t_net}_{parser.dataset_name}'


    train_file_path, train_file_label = read_path(save_path, train_fpath_label)
    test_file_path , test_file_label  = read_path(save_path, test_fpath_label)

    print(' ')


    # file_list1      = glob.glob(os.path.join(save_path, '*_ori.npy'))
    # file_list2      = glob.glob(os.path.join(save_path, '*_adv.npy'))
    # file_list3      = glob.glob(os.path.join(save_path, '*_noise.npy'))
    # file_list4      = glob.glob(os.path.join(save_path, '*_adv_deepfea.npy'))
    # file_list5      = glob.glob(os.path.join(save_path, '*_ori_deepfea.npy'))
    # file_list6      = glob.glob(os.path.join(save_path, '*_noise_deepfea.npy'))
    # file_list7      = glob.glob(os.path.join(save_path, '*_adv_fea.npy'))
    # file_list8      = glob.glob(os.path.join(save_path, '*_ori_fea.npy'))
    # file_list9      = glob.glob(os.path.join(save_path, '*_noise_fea.npy'))


    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal         =  load_data(get_data_path(test_file_path, '_ori_deepfea.npy'))
    uncerts_adv            =  load_data(get_data_path(test_file_path, '_adv_deepfea.npy'))
    uncerts_noisy          =  load_data(get_data_path(test_file_path, '_noise_deepfea.npy'))

    uncerts_normal_values  =  uncerts_normal.var(axis=0).mean(axis=0)
    uncerts_adv_values     =  uncerts_adv.var(axis=0).mean(axis=0)
    uncerts_noisy_values   =  uncerts_noisy.var(axis=0).mean(axis=0)

    print('Getting deep feature representations...')
    X_train_features       = get_data_path(train_file_path, '_ori_deepfea.npy')

    X_test_normal_features = get_data_path(test_file_path,  '_ori_deepfea.npy')

    X_test_noisy_features  = get_data_path(test_file_path,  '_noise_deepfea.npy')

    X_test_adv_features    = get_data_path(test_file_path,  '_adv_deepfea.npy')


    from sklearn.neighbors import KernelDensity
    class_inds = {}
    kdes       = {}
    for i in range(classes):
        class_inds[i] = np.where(np.array(train_file_label) == i)[0]
    X_train_features = load_data(X_train_features)
    for i in range(classes):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=1).fit(np.array(X_train_features)[class_inds[i]])
    print(kdes)

