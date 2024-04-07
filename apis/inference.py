import os
import os.path as osp
from operator import itemgetter

import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from ..datasets.pipelines import Compose
from ..models import build_recognizer


def init_recognizer(config,
                    checkpoint=None,
                    device=None,
                    use_frames=False):
    """Initialize a recognizer from config file.

    Args:
        config (str | :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str | None, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Default: None.
        device (str | :obj:`torch.device`): The desired device of returned
            tensor. Default: 'cuda:0'.
        use_frames (bool): Whether to use rawframes as input. Default:False.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if ((use_frames and config.dataset_type != 'RawframeDataset')
            or (not use_frames and config.dataset_type != 'VideoDataset')):
        input_type = 'rawframes' if use_frames else 'video'
        raise RuntimeError('input data type should be consist with the '
                           f'dataset type in config, but got input type '
                           f"'{input_type}' and dataset type "
                           f"'{config.dataset_type}'")
    # pretrained model is unnecessary since we directly load checkpoint later
    config.model.backbone.pretrained = None
    model = build_recognizer(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=device)
    model.cfg = config
    model.to(device)
    model.eval()
    return model

