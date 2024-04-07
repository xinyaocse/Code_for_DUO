import os
import torch
from PIL import Image
import torch.nn as nn
import scipy.io as sio
import torchvision.transforms as transforms


class TrainBase(object):
    def __init__(self, s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device, classes):
        self.MSE_loss = nn.MSELoss()
        cnt_per_cls = None
        if 'HMDB' in parser.dataset_name:
            cnt_per_cls = 2
        elif 'UCF' in parser.dataset_name:
            cnt_per_cls = 1
        else:
            print('Not sure which datatset is used')
            exit()
        self.all_cls = list(range(classes)) * cnt_per_cls
        self.all_cls_ori = list(range(classes)) * cnt_per_cls

        self.cls_num = classes * cnt_per_cls
        self.args = parser
        self.theta = torch.zeros(1)
        self.mask_I = torch.zeros(1)
        self.mask_F = torch.zeros(1)
        self.attack_method = parser.attack_method
        self.n_query = parser.n_query
        self.device = device
        self.s_db = s_db
        self.s_dl = s_dl
        self.t_db = t_db
        self.t_dl = t_dl
        self.t_dp = np.load(parser.t_net_list[0] + f'/mAP{parser.t_net_list[1]}_db_path.npy')
        self.max_p = parser.max_p
        s_net.eval()
        t_net.eval()
        self.s_net = build_net(s_net)
        self.t_net = build_net(t_net)
        parser.k = parser.k * 1e4  # k=16   parser.k = 16  spa =

        # parser.k   = 3*16*112*112   #
        self.kappa = self.args.kappa

        self.query_iter = query_iter
        # self.target_iter = target_iter
        self.target_iter = cycle(target_iter)
        self.save_path = parser.log_file
        os.makedirs(self.save_path, exist_ok=True)
        self.results = {}

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),  # Center
            transforms.ToTensor(),  # 0-255，顺序不能调换
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])

        self.target_map = float(parser.target_map)

        # sub_dataset_size = 40
        # tmp_idx = {}
        # new_t_db = []
        # new_t_dl = []
        # cnt    = {}
        # np.random.seed(2022)
        # for i in range(t_dl.size-1):
        #     tmp_i = t_dl[i]
        #     if tmp_idx.get(tmp_i) == None:
        #         tmp_idx[tmp_i]=[]
        #         tmp_idx[tmp_i].append(i)
        #         cnt[tmp_i] = 1
        #     else:
        #         tmp_idx[tmp_i].append(i)
        #         cnt[tmp_i] += 1
        # for i in range(classes):
        #     r_idx = np.random.choice(list(range(len(tmp_idx[i]))), sub_dataset_size, replace=False)
        #     random_idx = np.array(tmp_idx[i])[r_idx]
        #
        #     new_t_db.append(t_db[random_idx])
        #     new_t_dl.append(t_dl[random_idx])
        # new_t_db = [token for st in new_t_db for token in st]
        # new_t_dl = [token for st in new_t_dl for token in st]
        # new_t_db = np.array(new_t_db)
        # new_t_dl = np.array(new_t_dl)
        #
        # self.t_dl = new_t_dl
        # self.t_db = new_t_db

        print(' ')

    def save_dic(self, result, name):
        if not os.path.exists(os.path.dirname(os.path.join(self.save_path, name + '.mat'))):
            os.makedirs(os.path.dirname(os.path.join(self.save_path, name + '.mat')))
        sio.savemat(os.path.join(self.save_path, name + '.mat'), result)

    #加载视频
    def load_video(self, video_list):
        frames_list = []
        for i in range(16):
            # image = Image.open(self.args.data_root + "/" + video_list[i]).convert("RGB")
            image = Image.open(self.args.data_root + video_list[i]).convert("RGB")
            frame = self.transform(image)
            frames_list.append(frame.squeeze_(0))
        frames_array = torch.stack(frames_list, dim=0)
        frames_array = frames_array.transpose(1, 0).unsqueeze(0).to(self.device)
        frames_array = unnormalization(frames_array)  # 反归一化，将值变为0-255
        videopath = video_list[0].split('/')[1]
        return frames_array, videopath

    #通过路径加载视频
    def load_video_by_path(self, frames_dir):
        l_ = [file for file in os.listdir(os.path.join(self.args.data_root, frames_dir)) if
              file.endswith(".jpg") or file.endswith(".png")]
        l_.sort(key=lambda x: str(x[:-4]))
        frames_length = 16
        l = [frames_dir + '/' + l_[int(round(i * len(l_) / float(frames_length)))] for i in range(frames_length)]

        video = self.load_video(l)
        return video


class build_net(nn.Module):
    def __init__(self, net):
        super(build_net, self).__init__()  # 调用父类的init函数
        self.net = net

    def forward(self, x, input_norm=False):
        if input_norm:
            x = x
        else:
            x = x.clamp(0, 255)
            x = normalization(x)
        out = self.net(x)
        out = l2n(out)
        return out


def cycle(iterable):
    while True:
        for x in iterable:
            # yield相当于return加生成器，当程序再次进入这个函数时，从这个函数的yield语句的下一句开始执行的，即实现迭代。
            yield x


import math
import numpy as np


def compute_MAP(db_binary, db_label, test_binary, test_label):
    if torch.is_tensor(test_binary):
        test_binary = test_binary.cpu().numpy()
    if torch.is_tensor(test_label):
        test_label = test_label.cpu().numpy()
    if len(test_binary.shape) == 1:
        test_binary = test_binary[np.newaxis, :]

    AP = []
    Ns = np.array(range(1, db_binary.shape[0] + 1)).astype(np.float32)
    for i in range(test_binary.shape[0]):
        query_binary = test_binary[i]
        query_label = test_label[i]
        scores = np.dot(query_binary, db_binary.T)  # 计算该查询图片的score
        query_result = np.argsort(-scores)  # 将-scores中的元素从小到大排列，即按照score从大到小排列，提取其对应的index(索引)，然后输出到query_result
        correct = (query_label == db_label[query_result])  # 计算正样本数
        P = np.cumsum(correct, axis=0) / Ns  # P=前k个中有多少个正样本 / k  返回给定axis上的累计和，这里是行累加
        AP.append(np.sum(P * correct) / np.sum(correct))  # 计算该查询图片的AP
    MAP = np.mean(np.array(AP))  # 计算所有查询图片的AP平均值
    # return round(MAP,5)
    if MAP == math.nan:
        MAP = 0
        MAP = np.array(MAP)
    return MAP

#返回索引
def return_indx(db_binary, db_label, test_binary, test_label, target_idx, kappa=-1):
    if torch.is_tensor(test_binary):
        test_binary = test_binary.detach().cpu().numpy()  ##detach(): 返回一个新的Tensor，但返回的结果是没有梯度的。cpu():把gpu上的数据转到cpu上。numpy():将tensor格式转为numpy
    if torch.is_tensor(test_label):
        test_label = test_label.cpu().numpy()  # 将数据转换到cpu上和转换成numpy
    if len(test_binary.shape) == 1:  # 一维张量
        test_binary = test_binary[np.newaxis, :]  # 扩维
    if kappa == -1:
        kappa = db_binary.shape[0]  # 返回的是行数

    AP = []
    Ns = np.array(range(1, kappa + 1)).astype(np.float32)  # [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ]
    for i in range(test_binary.shape[0]):
        query_binary = test_binary[i]
        query_label = test_label[i]
        scores = np.dot(query_binary, db_binary.T)
        query_result = np.argsort(-scores)[:kappa]  # 获得 与查询视频相似的视频索引按照从大到小的顺序排序   query_result

        P = 0
        for j in range(kappa):  # j= 10 ,判断 query_result 和target_idx 的前10个有多少索引是相似的
            P = (len(set(target_idx[:j + 1]) & set(query_result[:j + 1])) / Ns[j])  # 计算当前的 presion
            AP.append(P)
        # correct = (target_idx == query_result)
        # P = np.cumsum(correct, axis=0) / Ns
        # AP.append(np.sum(P * correct) / np.sum(correct))
    MAP = np.mean(np.array(AP))  # 计算均值
    if math.isnan(MAP):
        MAP = 0

    return MAP


"""实现NDCG函数"""


def return_label(db_binary, db_label, test_binary, test_label, kappa=-1):
    if torch.is_tensor(test_binary):
        test_binary = test_binary.detach().cpu().numpy()  # detach(): 返回一个新的Tensor，但返回的结果是没有梯度的。cpu():把gpu上的数据转到cpu上。numpy():将tensor格式转为numpy
    if torch.is_tensor(test_label):  # 判断是不是张量
        test_label = test_label.cpu().numpy()  # 将数据转cpu和转numpy
    if len(test_binary.shape) == 1:
        test_binary = test_binary[np.newaxis, :]  # np.newaxis的功能:插入新维度，对test_binary 进行扩维
    if kappa == -1:  #
        kappa = db_binary.shape[0]  # kappa是db_binary的长度

    AP = []
    Ns = np.array(range(1, kappa + 1)).astype(np.float32)  # astype实现数据类型的转换
    for i in range(test_binary.shape[0]):
        query_binary = test_binary[i]
        query_label = test_label[i]
        scores = np.dot(query_binary, db_binary.T)  # 向量点积和矩阵乘法，返回一个矩阵,db_binary.T 对db_binary进行转置操作
        query_result = np.argsort(-scores)[:kappa]  # 按从大到小排序的索引
        correct = (query_label == db_label[query_result])
        P = np.cumsum(correct, axis=0) / Ns  # 按照 axis 轴进行累积求和,输出值中后一个数是前面所有数加上自己本身的和。
        AP.append(np.sum(P * correct) / np.sum(correct))
    MAP = np.mean(np.array(AP))  # 对所有元素求均值
    if math.isnan(MAP):  # nan是空
        MAP = 0

    # return round(MAP,5)
    return MAP, db_label[query_result], query_result


def find_diff(a, b, kappa):
    cnt = 0
    for i in range(kappa):
        if a[i] != b[i]:
            cnt += 1
    return cnt


from tqdm import tqdm


def inference(dataloader, net, device): #
    hashcodes = list()
    labels = list()
    path_list = list()
    # 进行validation，使用net.eval()切换到测试模式
    # 不会影响各层的gradient计算行为，不进行反向传播
    net.eval()
    # 停止gradient计算，从而节省了GPU算力和显存
    with torch.no_grad():
        for imgs, labels_, path in dataloader:  # Tqdm是一个快速，可扩展的Python进度条。tqdm(iterator)
            labels.append(labels_.view(labels_.size()[0], ).numpy())
            # print('imgs:', imgs.size())
            path_list.append(np.array([path[i][0] for i in range(16)]))
            hash_features = l2n(net(imgs.to(device)))  # 把像素信息传入网络得出预测结果
            hashcodes.append(hash_features.cpu().numpy())  # 把CUDA tensor格式的数据改成numpy
            continue
    # print hashcodes-threshold
    return np.concatenate(hashcodes), np.concatenate(labels), np.concatenate(path_list)


def l2n(x, eps=1e-6):
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, dim=0)
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)  # 保持维度


def perturbed_clamp(input, min=0, max=255): #扰动收敛
    output = normalization(unnormalization(input).clamp(min, max))
    return output


def unnormalization(delta):  # 反归一化
    MEAN = torch.tensor([[[[0.434]]], [[[0.405]]], [[[0.378]]]]).to(delta.device)
    STD = torch.tensor([[[[0.152]]], [[[0.149]]], [[[0.157]]]]).to(delta.device)
    clamp_delta = ((delta.data * STD + MEAN) * 255)

    return clamp_delta


from torchvision import transforms as T


def normalization(delta):  # 对delta进行归一化
    MEAN = torch.tensor([[[[0.434]]], [[[0.405]]], [[[0.378]]]]).to(delta.device)
    STD = torch.tensor([[[[0.152]]], [[[0.149]]], [[[0.157]]]]).to(delta.device)
    clamp_delta = (delta / 255 - MEAN) / STD

    # tr = T.Normalize(MEAN, STD)
    # clamp_delta = tr(delta)

    return clamp_delta


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def show_img(data, save_path='./', show=False):
    if data.shape[2] != 3:
        data = data.squeeze(dim=0).cpu().permute(1, 2, 0).numpy().astype('uint8')
    else:
        data = data.squeeze(dim=0).cpu().numpy().astype('uint8')

    plt.figure()  #
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data)
    if show:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def norm2(x):
    x = x.squeeze(dim=0)
    assert len(list(x.size())) == 4  # [c, t, h, w]
    x = x.permute(1, 0, 2, 3)  # [t, c, h, w]
    norm_vec = torch.sqrt(x.float().pow(2).sum(dim=[1, 2, 3])).view(1, 1, -1, 1, 1)  # [b, c, t, h, w]
    norm_vec += (norm_vec == 0).float() * 1e-8
    # print(norm_vec.squeeze())
    return norm_vec


def momentum_prior_step(x, g, lr):
    # adapted from Boosting Adversarial Attacks with Momentum, CVPR 2018
    return x + lr * g / norm2(g)
