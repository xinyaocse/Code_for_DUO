import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']


# 卷积块。定义了3*3卷积
# 该函数继承自nn网络中的3维卷积，这样做主要是为了方便，少写参数，参数由原来的6个变成了3个
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


# BasicBlock由两个3*3卷积组成
# 'resnet18', 'resnet34', 这两个网络使用基础版残差块BasicBlock
class BasicBlock(nn.Module):
    expansion = 1  # 对输出通道数的倍乘，这里不变

    # __init__用于对网络元素进行初始化，forward用于定义网络前向传播的规则

    # 基础版残差块，由两个叠加的3*3卷积组成
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)  # 结果向上取整
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # shortcut 支路

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # 卷积支路

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual   # 两条支路相加
        out = self.relu(out)

        return out


# 进阶版残差块 Bottleneck
class Bottleneck(nn.Module):
    expansion = 4  # 用来表征残差结构卷积核个数的变化
    # 与基础版的不同之处在于这里是三个卷积，分别是1x1,3x3,1x1,分别用来压缩维度，卷积处理，恢复维度。
    # inplane是输入的通道数，plane是输出的通道数，expansion是对输出通道数的倍乘，在basic中expansion是1，
    # 此时完全忽略expansion，输出的通道数就是plane，然而bottleneck要对通道数进行压缩，再放大，于是，plane不再代表输出的通道数，而是block内部压缩后的通道数，输出通道数变为plane*expansion。

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x   # shortcut 支路

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)   # 卷积支路

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 两条支路相加
        out = self.relu(out)

        return out


# 基础ResNet框架
class ResNet(nn.Module):

    def __init__(self, block, layers, shortcut_type='A'):  # A   layers=[2,2,2,2]
        self.inplanes = 64  # inplane是输入的通道数
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        ### above new
        # self.classifier=nn.Linear(512,num_classes) #
        # self.hashcoder=nn.Sequential(nn.Linear(512,hash_length),nn.Tanh()) #

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        # 第一个输入block是Bottleneck或BasicBlock类，第二个输入是该blocks的输出channel，第三个输入是每个blocks中包含多少个residual子结构
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print x.size()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # out = x.view(x.size(0), -1)
        # out = torch.squeeze(x)
        out = x.squeeze(-1).squeeze(-1)
        # x = self.fc(x)

        ### above new
        # c=self.classifier(x)
        # h=self.hashcoder(x)
        return out


# 各个残差网络
def resnet18(**kwargs):
    """Constructs a ResNet-18 network.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 network.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 network.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 network.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


class TemporalAvgPool(nn.Module):
    def __init__(self):
        super(TemporalAvgPool, self).__init__()
        self.filter = nn.AdaptiveAvgPool1d(1)  # 自适应一维平均池化，第一维只有1个值

    def forward(self, x):
        out = self.filter(x)
        out = torch.squeeze(out)  # 对数据的维度进行压缩，去掉维数都为1的维度，比如一行或一列
        return out

# 加载保存好的模型
def load_state(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith("module.")):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)




class Resnet18(nn.Module):
    """Constructs a (ResNet-18+Avg Pooling+Hashing ) network.
    """

    def __init__(self, classes):
        super(Resnet18, self).__init__()
        self.resnet = resnet18()
        #load_state(self.resnet, "./network/pretrain/resnet-18-kinetics.pth")  # 加载保存好的模型
        self.avgpooling = TemporalAvgPool()
        self.feature_layer = nn.Sequential(nn.Linear(512, classes))

    def forward(self, x):
        resnet_feature = self.resnet(x)
        avgpooling_feature = self.avgpooling(resnet_feature)
        feature = self.feature_layer(avgpooling_feature)

        return feature


class Resnet34(nn.Module):
    """Constructs a (ResNet-18+Avg Pooling+Hashing ) network.
    """

    def __init__(self, classes):
        super(Resnet34, self).__init__()
        self.resnet = resnet34()
        #load_state(self.resnet, "./network/pretrain/resnet-34-kinetics.pth")  # 加载保存好的模型
        self.avgpooling = TemporalAvgPool()
        self.feature_layer = nn.Sequential(nn.Linear(512, classes))

    def forward(self, x):
        resnet_feature = self.resnet(x)
        avgpooling_feature = self.avgpooling(resnet_feature)
        feature = self.feature_layer(avgpooling_feature)
        return feature

    # def forward1(self, x):
    #     resnet_feature = self.resnet(x)
    #     avgpooling_feature = self.avgpooling(resnet_feature)
    #
    #     return avgpooling_feature



class Resnet50(nn.Module):
    """Constructs a (ResNet-50+Avg Pooling+Hashing ) network.
    """

    def __init__(self, classes):
        super(Resnet50, self).__init__()
        self.resnet = resnet50()
        #load_state(self.resnet, "./network/pretrain/resnet-50-kinetics.pth")  # 加载保存好的模型
        self.avgpooling = TemporalAvgPool()
        self.feature_layer = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, classes))

    def forward(self, x):
        resnet_feature = self.resnet(x)
        avgpooling_feature = self.avgpooling(resnet_feature)
        feature = self.feature_layer(avgpooling_feature)
        return feature


class Resnet101(nn.Module):
    """Constructs a (ResNet-101+Avg Pooling+Hashing ) network.
    """

    def __init__(self, classes):
        super(Resnet101, self).__init__()
        self.resnet = resnet101()
        #load_state(self.resnet, "./network/pretrain/resnet-101-kinetics.pth")  # 加载保存好的模型
        self.avgpooling = TemporalAvgPool()
        self.feature_layer = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, classes))

    def forward(self, x):
        resnet_feature = self.resnet(x)
        avgpooling_feature = self.avgpooling(resnet_feature)
        feature = self.feature_layer(avgpooling_feature)
        return feature
