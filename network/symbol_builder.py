from mmaction.models import build_model
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, set_random_seed, load_checkpoint

import torch

model_chek = {
    'UCF101': {
        'TPN': {
            'ARCFACELOSS16frames': 'net_epoch30_mAP0.820575_ARCFACELOSS.pth'
        }
    },
    'HMDB51': {

    }

}


def dataset_cfg(opt):
    if 'mnt' in os.getcwd():
        data_root = '/mnt/nvme2n1/zhanyu/dataset/'
    else:
        data_root = '/root/public/dev8T/Dataset/'

    if opt.dataset_name == 'UCF101':
        root_folder = data_root + "UCF-101-frames/"
        train_fpath_label = "./DataProcess/UCF101/trainlist01.txt"
        test_fpath_label = "./DataProcess/UCF101/testlist01.txt"
        db_fpath_label = "./DataProcess/UCF101/trainlist01.txt"
        classes = 101
    elif opt.dataset_name == 'HMDB':
        root_folder = data_root + "hmdb51-frames/"
        train_fpath_label = "./DataProcess/HMDB51/hmdb51_split1_train.txt"
        test_fpath_label = "./DataProcess/HMDB51/hmdb51_split1_test.txt"
        db_fpath_label = "./DataProcess/HMDB51/hmdb51_split1_train.txt"
        classes = 51
    elif 'UCF101_' in opt.dataset_name:
        root_folder = data_root + "UCF-101-frames/"
        train_fpath_label = "./DataProcess/UCF101/trainlist01_{}.txt".format(str(opt.dataset_name.split('_')[-1]))
        test_fpath_label = "./DataProcess/UCF101/testlist01.txt"
        db_fpath_label = "./DataProcess/UCF101/trainlist01.txt"
        classes = 101
    elif 'HMDB_' in opt.dataset_name:
        root_folder = data_root + "hmdb51-frames/"
        train_fpath_label = "./DataProcess/HMDB51/hmdb51_split1_train_{}.txt".format(
            str(opt.dataset_name.split('_')[-1]))
        test_fpath_label = "./DataProcess/HMDB51/hmdb51_split1_test.txt"
        db_fpath_label = "./DataProcess/HMDB51/hmdb51_split1_train.txt"
        classes = 51
    else:
        print('dataset_name error')
        exit(0)

    return root_folder, train_fpath_label, test_fpath_label, db_fpath_label, classes


import os
import numpy as np


def net_cfg(network, embedding_dim, dataset_name, lossname, gpus, MAP, parser, device):
    cfg_options = {}

    d_ids = [gpus]
    # d_ids = [1]
    rec_root = 'configs/recognition/'
    if network.upper() in ['I3D', 'C3D', 'TPN', 'SLOWFAST', 'C3DFEA']:
        if network.upper() == 'I3D':
            config_rec = rec_root + 'i3d/i3d_r50_32x2x1_100e_ucf_rgb.py'
            config = rec_root + 'i3d/i3d_r50_32x2x1_100e_ucf_rgb.py'
            checkpoint_rec = 'work_dirs/i3d_r50_32x2x1_100e_ucf_split_1_rgb/epoch_100.pth'
        # if network.upper() == 'C3DFEA':
        #     config_rec = rec_root + 'c3d/c3d_fea_sports1m_16x1x1_45e_ucf101_rgb.py'
        #     config = rec_root + 'c3d/c3d_fea_sports1m_16x1x1_45e_ucf101_rgb.py'
        #     checkpoint_rec = 'work_dirs/c3d_fea_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth'
        if network.upper() == 'C3D':
            config_rec = rec_root + 'c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py'
            config = rec_root + 'c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py'
            checkpoint_rec = 'work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth'
        if network.upper() == 'TPN':
            config_rec = rec_root + 'tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py'
            config = rec_root + 'tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py'
            checkpoint_rec = 'work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf101_1_rgb/epoch_145.pth'
        if network.upper() == 'SLOWFAST':
            config_rec = rec_root + 'slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py'
            config = rec_root + 'slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py'
            checkpoint_rec = 'work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth'
        cfg = Config.fromfile(config)
        cfg_rec = Config.fromfile(config_rec)  # 以字典的形式返回预先写好的一个配置文件中的信息
        cfg_rec.merge_from_dict(cfg_options)
        cfg_rec.model['cls_head']['num_classes'] = embedding_dim

        net = build_model(
            cfg_rec.model, train_cfg=cfg_rec.train_cfg, test_cfg=cfg_rec.test_cfg)
        net = MMDataParallel(net, device_ids=d_ids)

    else:
        if network == 'Resnet18':
            from network.Resnet import Resnet18
            net = Resnet18(embedding_dim)
        elif network == 'Resnet34':
            from network.Resnet import Resnet34
            net = Resnet34(embedding_dim)
        net = torch.nn.DataParallel(net, device_ids=d_ids)


    # torch.cuda.set_device(1)
    checkpoint_path = './checkpoint/checkpoints_' + str(embedding_dim) + '/' + dataset_name + '/' + network + '/' + str(
        lossname) + '16frames'

    dict_path = os.path.join(checkpoint_path,
                             f'dat_{dataset_name}_{network}_{str(embedding_dim)}_{str(lossname)}_mAP{MAP}.pth')
    loadDict = torch.load(dict_path, map_location='cpu')
    net.load_state_dict(loadDict)

    db_binary = np.load(checkpoint_path + f'/mAP{MAP}_db_binary.npy')
    db_label = np.load(checkpoint_path + f'/mAP{MAP}_db_label.npy')
    test_binary = np.load(checkpoint_path + f'/mAP{MAP}_test_binary.npy')
    test_label = np.load(checkpoint_path + f'/mAP{MAP}_test_label.npy')

    if lossname != 'TRIPLETMARGINLOSS':
        parser.t_net_list = [checkpoint_path, MAP]
    net.eval()
    return net, db_binary, db_label, test_binary, test_label
