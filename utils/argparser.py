import argparse
import os
import socket
import logging


def set_logger(log_file='', debug_mode=False):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    handlers = [logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def printf(content):
    logging.info(content)

    print(content)


from mmcv import Config, DictAction

"""The parser is the default setting, parser generally refers to the process of converting text (strings) in a certain format into a certain data structure"""


def net_parser(parser):
    parser.add_argument('--dataset_name', default='UCF101', help='HMDB or UCF101 or JHMDB')
    parser.add_argument('--s_datascale', type=int, default=1111)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--num_frames', type=int, default=16, help='number of frames taken form a video')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--t_net', default='I3D',
                        choices=['Resnet34',
                                 'TPN',
                                 'I3D',
                                 'SlowFast'])
    parser.add_argument('--s_net', default='C3D', choices=['Resnet18', 'C3D', 'C3DFea'])
    parser.add_argument('--s_dim', type=int, default=512, choices=[256, 512, 768, 1024])
    parser.add_argument('--t_dim', type=int, default=768, choices=[768])
    parser.add_argument('--kappa', type=int, default=256)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--max_spa', type=int, default=3000)
    parser.add_argument('--gpus', type=int, default=5)
    parser.add_argument('--lossname', type=str, default='LIFTED',
                        choices=['TRIPLETMARGINLOSS',
                                 'ANGULARLOSS',
                                 'ARCFACELOSS',
                                 'MARGINLOSS',
                                 'MULTISIM',
                                 'CONTRASTIVE',
                                 'LIFTED'])  # new
    parser.add_argument('--debug-mode', type=bool, default=True,
                        help="print all setting for debugging.")
    parser.add_argument('--random-seed', type=int, default=2022,
                        help='random seed (default: 1)')
    return parser


def spa_parser(parser):
    parser.add_argument('--attack_method', type=str, default="HSA_SIMBA_SPA",
                        choices=[
                            'HSA_SIMBA_SPA',
                            'HSA_SIMBA_RAND',
                            'HSA_SIMBA_HEURISTIC',
                            'HSA_NES_HEURISTIC',
                            'SPA_TAR',
                            'MIM_TRAN',
                            'BIM_TRAN',
                            'FG_TAR',
                            'FG_TRAN',
                            'GEO_ATT'])
    parser.add_argument('--n_query', type=int, default=10,
                        help='how many query case in each categories')
    parser.add_argument('--path', default='pixel_k2n1nspagp', help="the exp save path")
    parser.add_argument('--G_path', default='group_k2n1nspa', help="the G_att exp save path")
    parser.add_argument('--group', type=str, default='G_spa', help='group spa attack')
    parser.add_argument('--trans', type=bool, default=False, help='trans between attack')
    parser.add_argument('--epsilon', type=float, default=0.078)
    parser.add_argument('--R', default=100, type=int)
    parser.add_argument('--T', default=0, type=int)
    parser.add_argument('--k', type=int, default=4, help='hyperparameters video perturbed pixels')
    parser.add_argument('--n', type=int, default=4, help='hyperparameters video perturbed frames')

    ###########update Theta I F paramters
    parser.add_argument('--maxIter_T', type=int, default=300, help='maxIter when updating noise Theta')
    parser.add_argument('--maxIter_I', type=int, default=1000, help='maxIter when updating mask I')
    parser.add_argument('--maxIter_F', type=int, default=300, help='maxIter when updating mask F')
    parser.add_argument('--maxIter_TG', type=int, default=300, help='maxIter when updating noise Theta')
    parser.add_argument('--maxIter_G', type=int, default=1000, help='maxIter when updating mask I')
    parser.add_argument('--maxIter_FG', type=int, default=300, help='maxIter when updating mask F')
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--sens', type=float, default=0.5)

    parser.add_argument('--tick_loss_T', type=int, default=100)
    parser.add_argument('--lr_T', type=float, default=0.01, help='initial learning rate for noist Theta')
    parser.add_argument('--lr_I', type=float, default=0.01, help='initial learning rate for mask I')
    parser.add_argument('--lr_F', type=float, default=0.01, help='initial learning rate for mask F')

    parser.add_argument('--tick_loss_e', type=int, default=400,
                        help='calculate loss per tick_loss_e iters while updating epsilon')
    ###########update F paramters

    ############trade off paramters
    # model hyper-parameters
    parser.add_argument('--lambdaL2', type=float, default=0, choices=['0', '1', '2'])
    parser.add_argument('--lambdaL3', type=float, default=0)

    parser.add_argument('--lambda2_upper_bound', type=float, default=0,
                        help='upper bound for lambda1')
    parser.add_argument('--lambda2_lower_bound', type=float, default=1e-3,
                        help='lower bound for lambda1')
    parser.add_argument('--lr_decay_step', type=int, default=50)
    parser.add_argument('--lr_decay_factor', type=float, default=0.95)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--maxIter_mm', type=int, default=1, help='maxIter for looping joint optimization')
    parser.add_argument('--trade_off', type=float, default=1e3)

    ###########search parameters

    parser.add_argument('--lambda2_search_times', type=int, default=1)

    ###########update I

    parser.add_argument('--cur_rho1', type=float, default=1e-2)
    parser.add_argument('--cur_rho2', type=float, default=1e-2)
    parser.add_argument('--cur_rho3', type=float, default=1e-7)
    parser.add_argument('--rho1_max', type=float, default=1e-2)  #1e-1
    parser.add_argument('--rho2_max', type=float, default=1e-2)  #1e-1
    parser.add_argument('--rho3_max', type=float, default=1e-7)  #1e-5
    parser.add_argument('--rho_increase_factor', type=float, default=1.01)
    parser.add_argument('--rho_increase_step', type=float, default=1)
    parser.add_argument('--cur_step', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--norm', type=str, default="l2", help="set logging file.")
    parser.add_argument('--t_net_list', type=list)
    parser.add_argument('--max_p', type=int, default=30)
    parser.add_argument('--target_map', type=float)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--n_q', type=int, default=10)
    parser.add_argument('--channels', type=int, default=3, help="set the channels size.")
    parser.add_argument('--mask_size', type=int, default=2, help="set mask size.")
    parser.add_argument('--imgsize', type=int, default=112, help="set imgsize.")
    parser.add_argument('--stride', type=int, default=1, help="set imgsize.")

    return parser


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Video Retrieval Attacking")

    parser = net_parser(parser)
    parser = spa_parser(parser)
    parser = parser.parse_args()

    parser.log_file = f'./exp/{parser.dataset_name}_{str(parser.s_datascale)}/t_net_{parser.t_net}_{parser.lossname}' \
                      f'/s_net_{parser.s_net}_{str(parser.s_dim)}/{parser.attack_method.upper()}'

    if parser.dataset_name == 'UCF101':
        parser.kappa = 256
    else:
        parser.kappa = 128
    return parser
