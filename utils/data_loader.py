import torch.utils.data as data
from attack.base import *
from PIL import Image


# 自定义数据集
class CustomDataset(data.Dataset):

    def __init__(self, root_folder, fpath_label, transform=None,
                 num_frames=32, device=None):  # fpath_label.txt: frames_dir video_label
        # 初始化文件路径或文件名列表
        f = open(fpath_label)
        l = f.read().splitlines()
        f.close()
        # print l
        # 对fpath与labels进行处理
        fpaths = list()
        labels = list()
        for item in l:
            v_id, label, path = item.split()
            fpaths.append(path.split('.')[0])
            labels.append(int(label))

        self.root_folder = root_folder
        self.fpaths = fpaths
        self.labels = labels
        self.label_size = len(self.labels)
        self.transform = transform
        self.num_frames = num_frames
        self.device = device # device configuration


    def __getitem__(self, index):
        # 返回图像和标签。按照索引读取每个元素的具体内容
        label = self.labels[index]
        ########## can use cv2 to process frames...#########
        path = self.fpaths[index]
        frames_dir = self.root_folder + path
        # print(frames_dir)
        l_ = [file for file in os.listdir(frames_dir) if file.endswith(".jpg") or file.endswith(".png")]
        l_.sort(key=lambda x: str(x[:-4]))
        frames_length = self.num_frames
        l = [l_[int(round(i * len(l_) / float(frames_length)))] for i in range(frames_length)]

        assert len(l) == self.num_frames
        frames_list = []
        # 3。 返回一个数据对(例如图像和标签)。
        frame_idx = []
        for i in range(frames_length):
            frame_idx.append(frames_dir + "/" + l[i])
            # 1. 从文件中读取一个数据(例如使用numpy.fromfile, PIL.Image.open)
            frame = Image.open(frames_dir + "/" + l[i]).convert("RGB")
            # 2. 预处理数据(例如:torchvision.Transform)。
            if self.transform is not None:
                # 如果transform不为None，则进行transform操作
                frame = self.transform(frame)
            frames_list.append(frame.squeeze_(0))
        frames_array = torch.stack(frames_list, dim=0)
        frames_array = frames_array.transpose(1, 0)

        label = torch.tensor(label)
        frames = frames_array.clone().detach().to(self.device)

        frames = unnormalization(frames)


        # for i in range(16):
        #     show_img(frames[:,i])
        return frames, label,[path+'/'+i for i in l]

    def __len__(self):
        # 返回所有图片的数量，即数据集长度
        return len(self.fpaths)
