import os
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from dataset.data import iterator_factory
from utils.train import metric
from utils.train.model import model
from utils.train.lr_scheduler import MultiFactorScheduler


def train_model(sym_net, parser):

    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    # data iterator
    iter_seed = torch.initial_seed()  \
                + (torch.distributed.get_rank() * 10 if parser.distributed else 100) \
                + max(0, parser.resume_epoch) * 100
    train_iter, eval_iter = iterator_factory.creat(name=parser.dataset,
                                                   batch_size=parser.batch_size,
                                                   clip_length=parser.clip_length,
                                                   train_interval=parser.train_frame_interval,
                                                   val_interval=parser.val_frame_interval,
                                                   mean=parser.datacfg['input_conf']['mean'],
                                                   std=parser.datacfg['input_conf']['std'],
                                                   seed=iter_seed,
                                                   data_root=parser.data_root)
    print(len(train_iter))
    print(len(eval_iter))
    # wapper (dynamic model)
    net = model(net=sym_net,
                criterion=torch.nn.BCELoss().cuda(),
                model_prefix=parser.model_prefix,
                step_callback_freq=10,
                save_checkpoint_freq=parser.save_frequency,
                opt_batch_size=parser.batch_size, # optional
                dataset=parser.dataset,  # dataset name
                hash_bit=parser.hash_bit,
                )


    net.net = torch.nn.DataParallel(net.net,device_ids=parser.device_ids).cuda()

    optimizer = torch.optim.SGD(net.net.parameters(), lr=parser.lr_base, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parser.lr_base)

    net.fit(train_iter=train_iter,
            eval_iter=eval_iter,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            epoch_start=0,
            epoch_end  =160)

