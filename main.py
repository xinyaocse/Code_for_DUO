"""
The codes are from DUO's work
Pixel_Sparse_AE_attack the is code of the pixel sparse adversarial AE attack method proposed by our previous work for the video retrieval system,
Group_Sparse_AE_attack the is code of the new attack method proposed by our current work to improve the query efficiency
"""

import json

from torch.distributions import transforms

from network.symbol_builder import dataset_cfg, net_cfg
from attack.HEUR_ATT import HEUR_attack
from attack.Pixel_Sparse_AE_attack import Pixel_Sparse_AE_attack
from attack.Group_Sparse_AE_attack import Group_Sparse_AE_attack


from utils.data_loader import *

"""Load the data"""


def load_data(root_folder, fpath_label, batch_size, shuffle=True, num_workers=16, train=False, num_frames=32,
              drop_last=True, device=None):
    if train:
        transform = transforms.Compose([
            # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),  #
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])
    else:

        transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])

    data_ = CustomDataset(root_folder=root_folder,
                          fpath_label=fpath_label,
                          transform=transform,
                          num_frames=num_frames, device=device)

    # torch.utils.data.DataLoader

    loader_ = data.DataLoader(
        dataset=data_,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=shuffle,  # shuffle
        num_workers=num_workers,
        drop_last=drop_last
    )

    return loader_



def cycle(iterable):
    while True:
        for x in iterable:
            yield x


from utils.argparser import *
from pathlib import Path
from utils.sort import find_dict
import datetime


def attack(parser):
    if parser.dataset_name == 'UCF101':
        parser.kappa = 256
    else:
        parser.kappa = 128
    set_logger(log_file=parser.log_file + f'/iterval_{str(parser.interval)}_n_{str(parser.n)}_k_{str(parser.k)}_'
                                          f'maxp_{str(parser.max_p)}_kappa_{str(parser.kappa)}'
                                          f'_{parser.attack_method.upper()}'
                                          f'.log', debug_mode=parser.debug_mode)
    # print(parser.log_file)
    logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))
    logging.info("Start training with args:\n" + json.dumps(vars(parser), indent=4, sort_keys=True))

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(parser.gpus)

    torch.cuda.set_device(parser.gpus)

    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(parser.random_seed)
    torch.cuda.manual_seed(parser.random_seed)

    root_folder, train_fpath_label, test_fpath_label, db_fpath_label, classes \
        = dataset_cfg(parser)

    parser.data_root = root_folder
    parser.network = parser.s_net

    s_map = find_dict(dataset=parser.dataset_name + '_' + str(parser.s_datascale), model_name=parser.s_net,
                      loss_name='TRIPLETMARGINLOSS'.upper() + '16frames', embedding_size=parser.s_dim)
    t_map = find_dict(dataset=parser.dataset_name, model_name=parser.t_net,
                      loss_name=parser.lossname.upper() + '16frames', embedding_size=parser.t_dim)

    parser.target_map = t_map
    device = torch.device(f"cuda:{parser.gpus}" if torch.cuda.is_available() else "cpu")  # device configuration

    s_net, s_db, s_dl, _, _ = net_cfg(parser.s_net, parser.s_dim, parser.dataset_name + f'_{str(parser.s_datascale)}',
                                      'TRIPLETMARGINLOSS', parser.gpus, MAP=s_map, parser=parser, device=device)
    t_net, t_db, t_dl, _, _ = net_cfg(parser.t_net, parser.t_dim, parser.dataset_name, parser.lossname, parser.gpus,
                                      MAP=t_map, parser=parser, device=device)




    dataset = np.load(f'./dataset/{parser.t_net}_{parser.dataset_name}_video_list_path.npy', allow_pickle=True)



    if parser.attack_method in ['HSA_SIMBA_SPA', 'HSA_SIMBA_RAND', 'HSA_SIMBA_HEURISTIC']:
        if parser.group == 'G_spa':
            hsab_attack = Group_Sparse_AE_attack(s_net, t_net, dataset, dataset, s_db, s_dl, t_db, t_dl, parser, device, classes)
            hsab_attack.attack()
        elif parser.group == "P_spa":
            hsa_attack = Pixel_Sparse_AE_attack(s_net, t_net, dataset, dataset, s_db, s_dl, t_db, t_dl, parser, device, classes)
            hsa_attack.attack()



if __name__ == "__main__":
    # for i in ['Resnet18','C3D']:
    parser = get_parser()
    # parser.s_net = i
    parser.log_file = f'./exp/{parser.path}/{parser.dataset_name}_{str(parser.s_datascale)}/t_net_{parser.t_net}_{parser.lossname}' \
                      f'/s_net_{parser.s_net}_{str(parser.s_dim)}/{parser.attack_method.upper()}'
    printf(parser.log_file)
    printf(f"{parser.group}")
    attack(parser)
