import os
import logging

import torch

from dataset.data import video_sampler as sampler
from dataset.data import video_transforms as transforms
from dataset.data.video_iterator import VideoIter


def generate_list(parser, pathSet):
    target_path, target_label, query_path, query_label = pathSet

    with open(os.path.join('./dataset', parser.dataset, 'raw', 'list_cvt', 'querylist01.txt'), 'w+') as f:
        for i in range(len(query_path)):
            f.write(str(i) + ' ' + str(query_label[i]) + ' ' + str(query_path[i]) + '\n')

    with open(os.path.join('./dataset', parser.dataset, 'raw', 'list_cvt', 'targetlist01.txt'), 'w+') as f:
        for i in range(len(target_path)):
            f.write(str(i) + ' ' + str(target_label[i]) + ' ' + str(target_path[i]) + '\n')


def get_hmdb51(data_root=None,
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=0,
               pathSet=0,
               parser=None, **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
        clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed + 0))
    generate_list(parser, pathSet)
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw-data'),
                      txt_list=os.path.join('./dataset/HMDB51', 'raw', 'list_cvt', 'querylist01.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                          transforms.RandomScale(make_square=True,
                                                 aspect_ratio=[0.8, 1. / 0.8],
                                                 slen=[224, 288]),
                          transforms.Resize((128, 171)),
                          transforms.RandomCrop((112, 112)),  # insert a resize if needed
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomHLS(vars=[15, 35, 25]),
                          transforms.ToTensor(),
                          normalize,
                      ],
                          aug_seed=(seed + 1)),
                      name='train',
                      shuffle_list_seed=(seed + 2),
                      )

    val_sampler = sampler.SequentialSampling(num=clip_length,
                                             interval=val_interval,
                                             fix_cursor=True,
                                             shuffle=True)
    val = VideoIter(
        video_prefix=os.path.join(data_root, 'raw-data'),
        txt_list=os.path.join('./dataset/HMDB51', 'raw', 'list_cvt', 'targetlist01.txt'),
        sampler=val_sampler,
        force_color=True,
        video_transform=transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            normalize,
        ]),
        name='test',
    )

    return (train, val)


def get_ucf101(data_root=None,
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=0,
               pathSet=0,
               parser=None, **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format(clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    generate_list(parser, pathSet)

    train = VideoIter(video_prefix=os.path.join(data_root, 'raw-data'),
                      txt_list=os.path.join('./dataset/UCF101', 'raw', 'list_cvt', 'querylist01.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                          transforms.Resize((128, 171)),
                          transforms.CenterCrop((112, 112)),
                          transforms.ToTensor(),
                          normalize,
                      ]),
                      name='test',
                      )

    val_sampler = sampler.SequentialSampling(num=clip_length,
                                             interval=val_interval,
                                             fix_cursor=True,
                                             shuffle=True)
    val = VideoIter(video_prefix=os.path.join(data_root, 'raw-data'),
                    txt_list=os.path.join('./dataset/UCF101', 'raw', 'list_cvt', 'targetlist01.txt'),
                    sampler=val_sampler,
                    force_color=True,
                    video_transform=transforms.Compose([
                        transforms.Resize((128, 171)),
                        transforms.CenterCrop((112, 112)),
                        transforms.ToTensor(),
                        normalize,
                    ]),
                    name='test',
                    )

    return (train, val)


def get_kinetics(data_root='./dataset/Kinetics',
                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
        clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed + 0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'train_avi-x256'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train_w-missed-v1_avi.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                          transforms.RandomScale(make_square=True,
                                                 aspect_ratio=[0.8, 1. / 0.8],
                                                 slen=[224, 288]),
                          transforms.RandomCrop((224, 224)),  # insert a resize if needed
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomHLS(vars=[15, 35, 25]),
                          transforms.ToTensor(),
                          normalize,
                      ],
                          aug_seed=(seed + 1)),
                      name='train',
                      shuffle_list_seed=(seed + 2),
                      )

    val_sampler = sampler.SequentialSampling(num=clip_length,
                                             interval=val_interval,
                                             fix_cursor=True,
                                             shuffle=True)
    val = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_avi-x256'),
                    txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_w-missed-v1_avi.txt'),
                    sampler=val_sampler,
                    force_color=True,
                    video_transform=transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ]),
                    name='test',
                    )
    return (train, val)


def creat(name, batch_size, num_workers=16, **kwargs):
    if name.upper() == 'UCF101':
        train, val = get_ucf101(**kwargs)
    elif name.upper() == 'JHMDB':
        train, val = get_hmdb51(**kwargs)
    elif name.upper() == 'KINETICS':
        train, val = get_kinetics(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))

    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val,
                                             batch_size=batch_size * kwargs['target_number'], shuffle=False,
                                             num_workers=num_workers, pin_memory=True, drop_last=True)

    return (train_loader, val_loader)
