import os
import time

import numpy as np
import torch
from attack.save_result import *
from attack.compute_statistic import *
from torch.autograd import Variable
import torch
from attack.base import compute_MAP, TrainBase, return_label
import logging
from utils.argparser import *
from numpy import *
import math
from attack.base import *


class HSA_attack(TrainBase):
    def __init__(self, s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device, classes):
        super(HSA_attack, self).__init__(s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device,
                                         classes)
        pass

    def attack(self):
        global que_tarlabel_tnet, i
        cnt = 0
        self.APo2o = []  # 原始视频在原始标签上面的精度
        self.APo2t = []  # 原始视频在目标标签上的精度
        self.t_mapMean = []  # target map的均值
        self.spaMean = []  # 稀疏性的均值
        datalen = self.query_iter.shape[0] if self.query_iter.shape[0] < 6 else 5  # .shape[0]是行数
        for i_batch in range(datalen):
            i_batch_data = self.query_iter[i_batch]  # i_BATCH 第i批

            self.query_label = torch.tensor([int(i_batch_data[1])])  # i_batch_data[1] = 37 生成一个大小为37的张量
            # print(self.query_label)
            self.query_path = i_batch_data[2][0][:-15]  # 返回i_batch_data的查询路径
            self.query_path_list = i_batch_data[2]  # 返回i_batch_data的查询路径列表

            self.target_label = torch.tensor([int(i_batch_data[3])])  # 生成一个大小为37的张量
            self.target_path = i_batch_data[4][0][:-15]  # 返回i_batch_data的目标路径
            self.target_path_list = i_batch_data[4]  # 返回i_batch_data的目标路径列表

            self.query, _ = self.load_video(self.query_path_list)  # 加载查询视频路径列表
            self.target, _ = self.load_video(self.target_path_list)  # 加载目标 视频路径列表

            # que_retlabel_tnet:请求视频在目标模型返回的标签  que_retindx_tnet：请求视频在目标模型返回的索引
            MAPo2o, que_retlabel_tnet, que_retindx_tnet = return_label(self.t_db,
                                                                       self.t_dl,
                                                                       self.t_net(self.query),
                                                                       self.query_label,
                                                                       self.kappa)

            # tar_retlabel_tnet:目标视频在目标模型返回的标签  tar_retindx_tnet：目标视频在目标模型返回的索引
            MAPo2t, tar_retlabel_tnet, self.tar_retindx_tnet \
                = return_label(self.t_db,
                               self.t_dl,
                               self.t_net(self.target),
                               self.query_label,
                               self.kappa)

            # t_net(self.query)返回查询视频在目标模型上面的一个特征
            loss = return_indx(self.t_db,
                               self.t_dl,
                               self.t_net(self.query).detach(),
                               self.target_label,
                               self.tar_retindx_tnet,
                               self.kappa)

            cnt += 1
            self.MAPo2t = MAPo2t
            self.MAPo2o = MAPo2o

            printf(f'cnt:{cnt}  '
                   f'Successful exmaple oriPath:{self.query_path}({self.query_label.item()}) '
                   f'MAPo2o:{MAPo2o:.4f} '
                   f'MAPo2t{MAPo2t:.4f}  '
                   f'tarPath:{self.target_path}({self.target_label.item()}) '
                   f'{loss:.4f}')

            self.attack_batch(self.query, i_batch)

    def attack_batch(self, query, i_batch):
        current_time = time.time()

        global result, adv_imgs

        self.tar_fea_snet = self.s_net(self.target).detach()  # 代理模型上目标视频的特征
        self.tar_fea_tnet = self.t_net(self.target).detach()  # 目标模型上目标视频的特征
        self.que_fea_tnet = self.t_net(query).detach()  # 请求视频在目标模型上的特征

        # tar_retindx:目标视频的检索列表，MAPt2t_tnet是目标视频到目标标签的一个MAP值
        MAPt2t_tnet, m1, self.tar_retindx = return_label(self.t_db,
                                                         self.t_dl,
                                                         self.tar_fea_tnet,
                                                         self.target_label,
                                                         self.kappa)
        # que_retindx：请求视频的检索列表，MAPo2o_tnet是原始标签到查询视频特征的一个MAP值
        MAPo2o_tnet, m2, self.que_retindx = return_label(self.t_db,
                                                         self.t_dl,
                                                         self.que_fea_tnet,
                                                         self.query_label,
                                                         self.kappa)
        # q2t_retindx是请求视频在目标标签下的一个检索列表，MAPo2t_tnet原始视频在目标标签上的一个map值
        MAPo2t_tnet, m3, self.q2t_retindx = return_label(self.t_db,
                                                         self.t_dl,
                                                         self.que_fea_tnet,
                                                         self.target_label,
                                                         self.kappa)

        PER_list = []
        SPA_list = []
        PROBS_list = []
        PREo2o_list = []
        PREo2t_list = []

        if 'HSA_SIMBA' in self.attack_method:
            if 'HSA_SIMBA_SPA' == self.attack_method:
                iteration_num = 1
            else:
                iteration_num = 1
            for rr in range(iteration_num):  # 构建保存路劲，迭代次数_k值_n值_最大扰动的值_步数值_原始视频_目标视频
                path = f'iterval_{str(int(self.args.interval))}_' \
                       f'n_{str(int(self.args.n))}_' \
                       f'k_{str(int(self.args.k / 1e4))}_' \
                       f'maxp_{str(int(self.max_p))}_' \
                       f'kappa_{str(int(self.kappa))}_' \
                       f'ori_{self.query_path.split("/")[-2]}_' \
                       f'tar_{self.target_path.split("/")[-2]}_{str(rr)}'
                # if f'{self.target_path.split("/")[-2]}' not in \
                #     'ori_Ukigumo_Kata_from_Hasegawa_Eishin_Ryu_draw_sword_f_nm_np1_le_med_1_tar_Takeda_Ryu_Iaido_draw_sword_f_nm_np1_fr_med_2_1':
                #     continue
                # if os.path.exists(os.path.join(self.save_path, path + '.mat')):
                #     continue

                result = self.HSA_SIMBA(query)  # 进行查询攻击，HSA_SIMBA是一种图像检索方法

                self.save_dic(result, path)  # 保存结果

                PER_list.append(result['Per_item'])
                SPA_list.append(result['Spa_item'])
                PROBS_list.append(result['Probs_item'])
                PREo2o_list.append(result['Preo2o_item'])
                PREo2t_list.append(result['Preo2t_item'])
                printf(f'i_batch {rr} '
                       f'SpaMean: {mean(SPA_list):.4f}  '
                       f'PerMean: {mean(PER_list):.4f}  '
                       f'ProbsMean: {mean(PROBS_list):.4f}  '
                       f'Preo2oMean: {mean(PREo2o_list):.4f} '
                       f'Preo2tMean: {mean(PREo2t_list):.4f}')

    # 对应论文里面的Vanilla 攻击，这个攻击不是最新的，可能要对照最新的攻击
    def heuristic_attack(self, spa_ratio, key_frame_num):  # spa_raio 稀疏率，key_frame_num 关键帧数量
        # spa_ratio   = 0.2
        from attack.ATT_FUNC import SpectralResidual, get_cv2_func
        cv2_func = get_cv2_func(0)
        spatial_mask = []
        for i in range(16):
            this_mask = SpectralResidual(cv2_func, self.query.cpu()[0, :, i], spa_ratio)
            spatial_mask.append(this_mask)
        spatial_mask = torch.from_numpy(np.array(spatial_mask)).unsqueeze(0).permute(0, 4, 1, 2,
                                                                                     3).cuda()  # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        advNoise = (self.target - self.query)  # 计算对抗噪声
        all_frames = [i for i in range(16)]
        query = 0
        score_dict = {}
        tar_probs, _, _, _ = self.get_probs(self.target, return_loss12=True)  # 获取到目标视频的损失函数
        que_probs, _, _, _ = self.get_probs(self.query, return_loss12=True)  # 获取到请求视频的损失函数

        for i in all_frames:
            tmp_frames = [_ for _ in all_frames if _ != i]
            MASK = torch.zeros_like(self.query).cuda()
            MASK[0, :, tmp_frames, :, :] = 1
            MASK = MASK * spatial_mask  # MASK是掩码
            adv_video = advNoise * MASK + self.query  # 别人的攻击的算法
            adv_probs, _, _, _ = self.get_probs(adv_video, return_loss12=True)  # 对抗样本的损失函数
            score_dict[i] = adv_probs
        sorted_items = sorted(score_dict.items(), key=lambda x: -x[1], reverse=True)
        key_frame_indx = [sorted_items[i][0] for i in range(key_frame_num)]  # 关键帧的索引
        MASK = torch.zeros_like(self.query).cuda()
        MASK[0, :, key_frame_indx, :, :] = 1
        MASK = MASK * spatial_mask
        return MASK

    def HSA_SIMBA(self, query):
        global trans, group_spa, transability, indices, prev_probs_test, left_probs, right_probs, theta_test, mask_I, mask_F, expand_dims, adv_retindx, max_queries, theta_right, adv_right, advMAPt2t, Preo2o, Preo2t, interval, max_queries_iterval
        max_iters = int(1e5)  # 最大迭代次数
        log_every = 10  # 每10次迭代，保存实验结果
        pixel_attack = True  # 像素攻击
        transability = False  # 可转移性
        group_spa = False  # 组稀疏
        query_size = query.size(4)  # 一个视频帧的高 = 112  query数据形式为（batch_size,channels,frames_number,width=112,height=112）
        query_frame = query.size(2)  # 视频的帧数     = 16
        batch_size = query.size(0)  # batch_size  = 1
        dims_intra = query_size  # 查询大小
        dims_inter = query_frame  # 查询帧数
        order = 'rand'

        # 如果是组稀疏的话，这地方可能也要改
        if order == 'rand':
            expand_dims = dims_intra
        n_dims = 3 * dims_inter * expand_dims * expand_dims  # 向量维度为3*16*112*112
        x = torch.zeros(batch_size, n_dims)  # 构建一个维度为1*3*16*112*112的向量

        probs1 = np.zeros((max_iters))  # 记录损失值
        probs2 = np.zeros((max_iters))  # 记录损失值
        probs = np.zeros((max_iters))  # 记录损失值
        queries = np.zeros((max_iters))  # 记录每一次迭代的请求次数
        cntSucc = np.zeros((max_iters))  # 判断当此加入的扰动是否有效，如果有效则设置为1，无效则设置为0
        advMapo2tlist = []  # map平均值列表
        Preo2tList = []  # 查询视频在目标标签上面的精度
        Preo2oList = []  # 查询视频在查询标签上面的精度
        max_p = self.max_p

        # 在目标模型上训练之后讲对抗视频放在目标模型是进行测值
        if transability:
            adv_video = query.clone()  # 这一行代码将adv_video初始化为query的副本，即将adv_video设置为与query相同。
            theta, mask_F, mask_I = self.sparse_attack(adv_video)
            prev_probs_test, prev_probs_test1, prev_probs_test2, _ = self.get_probs(adv_video, return_loss12=True)
        else:
            adv_video = query.clone()  # 将查询的结果赋值给对抗样本视频
            theta, mask_F, mask_I = self.sparse_attack(adv_video)  # 先在替代模型上面运行SPA，在进行查询攻击

            # 计算没有添加稀疏扰动时视频的概率分布,其中prev_probs_test是完整的预测结果。
            prev_probs_test, prev_probs_test1, prev_probs_test2, _ = self.get_probs(adv_video, return_loss12=True)

        if pixel_attack:
            trans = lambda z: z  # 把一维向量还原成  视频的形状

        # adv_video = query.clone()  # 将查询的结果赋值给对抗样本视频
        # theta, mask_F, mask_I = self.sparse_attack(adv_video)  # 先在替代模型上面运行SPA，在进行查询攻击
        # prev_probs_test, prev_probs_test1, prev_probs_test2, _ = self.get_probs(adv_video,
        #                                                                         return_loss12=True)  # 根据NDCG损失函数计算

        # 将每个特征向量表示为一组“左向量”和“右向量”的组合，并使用这些向量的内积来计算相似度
        left_probs = 10  # 左侧特征向量的伪概率值
        right_probs = 10  # 右侧特征向量的伪概率值
        mask_F = torch.ones_like(query)  # 生成与查询视频向量形状相同的全1张量
        mask_I = torch.ones_like(query)  # 生成与查询视频向量形状相同的全1张量

        max_queries = int(self.args.max_queries)  # 最大次数的查询量
        interval = int(self.args.interval)  # 两个事件之间的时间间隔

        if self.args.attack_method == 'HSA_SIMBA_RAND':  # 随机选点
            # 可替换为查找文件确定 k。
            printf(f'Running Rand Select Mec...........')
            max_queries_iterval = int(max_queries / interval)  # 在间隔为2下面的最大查询次数
            indices = torch.randperm(
                3 * dims_inter * dims_intra * dims_intra)  # 生成一个长度为 3 * dims_inter * dims_intra * dims_intra 的随机排列索引，这个索引可以用于打乱张量中的元素顺序
            printf(f'Fininsh Random attack max_queries:{str(int(max_queries))} ...........')

        if self.args.attack_method == 'HSA_SIMBA_HEURISTIC':  # 显著区域的关键帧选点
            printf(f'Running Heuristic indics Select Mec  ...........')
            spa_ratio, key_frame = float((self.args.k / 602112)), self.args.n

            mask_H = self.heuristic_attack(spa_ratio, key_frame)  # 进行启发式攻击，对应论文的香草攻击
            return_data, order = torch.sort(torch.abs((mask_H).reshape(-1)), descending=True)
            indices = order.clone()  # 使用clone进行深拷贝
            max_queries_iterval = int(torch.sum(torch.abs(mask_H)).item())
            printf(f'Fininsh Heuristic indics Select Mec with k:{str(int(self.args.k))} n:{str(int(self.args.n))}'
                   f' SPA:{str(max_queries_iterval)}...........')

        cnt = 0
        for mm in range(int(interval)):
            if self.args.attack_method == 'HSA_SIMBA_SPA':  # 稀疏算法选点
                printf(f'Running Spare Mec on {str(int(mm))}...........')
                theta, mask_F, mask_I = self.sparse_attack(adv_video)  # 对视频进行稀疏性攻击，返回扰动的扰动量、扰动帧、扰动的像素点

                # theta, mask_F, mask_G = self.sparse_attack(adv_video)  # 对视频进行稀疏性攻击，返回扰动的扰动量、扰动帧、扰动的像素块

                # 可以将这段代码理解为对视频中每个时间步和空间位置的稀疏扰动进行排序，以便后续选择其中的一部分进行加入，形成最终的稀疏扰动。
                # 将结果展开为一维张量，取绝对值并降序排序。
                return_data, order = torch.sort(torch.abs((mask_F * mask_I * theta).reshape(-1)), descending=True)
                indices = order.clone()  # 用于创建一个张量的深度拷贝，即创建一个新的张量，并将源张量的值复制到新张量中。
                max_queries_iterval = int(torch.sum(mask_F * mask_I).item())  # max_queries_iterval是本次迭代的最大查询次数

                printf(f'Fininsh Spare Mec with k:{str(int(self.args.k))} n:{str(int(self.args.n))}'
                       f' SPA:{str(int(torch.sum(torch.abs(mask_F * mask_I)).item()))}...........')

            for k in range(max_queries_iterval):
                cnt += 1
                ################################################################################################################
                dim = indices[k]  # 其中indices是一个随机排列的整数序列，用于随机选择要扰动的位置。
                diff = torch.zeros(1, n_dims)  # 1 * 3 * 16 * 112 * 112
                diff[:, dim] = max_p  # 是在diff张量的第dim列上更新max_p的值，从而更新了diff张量。
                left_vec = x + diff  # 将x向量中第dim个元素加上max_p，其余元素不变。
                right_vec = x - diff  # 将x向量中第dim个元素减去max_p，其余元素不变。

                # try negative directions

                # 首先将left_vec通过expand_vector转换为和theta相同的形状，然后通过trans函数将theta_left的维度顺序从(batch_size, dims_inter, expand_dims, expand_dims) 转化为 (batch_size, expand_dims, expand_dims, dims_inter)。
                theta_left = trans(self.expand_vector(left_vec, expand_dims))  # 将向量还原成视频
                # 先将左偏扰动加到原始视频上得到新的视频，再将其和右偏扰动相比较，取元素值较大的那个，最后将得到的视频的像素值限制在0到255之间。
                adv_left = torch.max(torch.min(theta_left + adv_video, query + max_p), query - max_p).clamp(0, 255)

                # 输出是预测的分类概率结果left_probs以及在两个层面的中间结果left_probs1和left_probs2，用于计算loss。
                left_probs, left_probs1, left_probs2, _ = self.get_probs(adv_left, return_loss12=True)

                queries_k = 1
                # 这段代码是一个比较，判断稀疏扰动后的视频经过模型后的分类概率是否比没有扰动前的视频分类概率更小。说明经过扰动后的视频被模型分类为了另外一个类别，因此这个扰动是有效的。
                if left_probs < prev_probs_test:
                    x[0, dim] = max_p  # 这里的 x[0, dim] = max_p 表示将第一个视频的 x 向量中的第 dim 个元素的值设置为 max_p
                    prev_probs_test = left_probs
                    cntSucc[cnt] = 1  # 这行代码的作用是将cntSucc数组在cnt位置的值设为1，表示进行cnt次攻击的成功次数为1。
                    adv_video = adv_left.clone()
                else:
                    # try positive directions
                    # 首先将right_vec通过expand_vector转换为和theta相同的形状，然后通过trans函数将theta_right的维度顺序从(batch_size, dims_inter, expand_dims, expand_dims) 转化为 (batch_size, expand_dims, expand_dims, dims_inter)。
                    theta_right = trans(self.expand_vector(right_vec, expand_dims)).clone()
                    queries_k = 2

                    # 怀疑这里有问题！！！！！！！！
                    adv_right = torch.max(torch.min(theta_left + adv_video, query + max_p), query - max_p).clamp(0, 255)
                    # adv_right = torch.max(torch.min(theta_right + adv_video, query + max_p), query - max_p).clamp(0, 255)
                    right_probs, right_probs1, right_probs2, _ = self.get_probs(adv_right, return_loss12=True)

                if right_probs < prev_probs_test:
                    # 先将右偏扰动加到原始视频上得到新的视频，再将其和右偏扰动相比较，取元素值较大的那个，最后将得到的视频的像素值限制在0到255之间。
                    adv_right = torch.max(torch.min(theta_right + adv_video, query + max_p), query - max_p).clamp(0,
                                                                                                                  255)
                    x[0, dim] = -max_p
                    prev_probs_test = right_probs
                    adv_video = adv_right.clone()
                    cntSucc[cnt] = 1

                probs2[cnt] = prev_probs_test2
                probs1[cnt] = prev_probs_test1
                probs[cnt] = prev_probs_test
                queries[cnt] = queries_k  # 成功进行攻击的像素位置进行的查询次数

                # if int(torch.sum(torch.abs((adv_video - query)) > 0).item()) > max_SPA:
                #     break
                # if sum(queries) > max_queries:
                #     break

                if k % 100 == 0:
                    # 这段代码是将x向量进行扩展，然后通过trans函数进行变换，得到theta_test向量。
                    theta_test = trans(self.expand_vector(x, expand_dims)).clone()
                    # 向原始视频帧添加扰动来创建对抗视频来执行视频攻击
                    adv_test = torch.max(torch.min(theta_test + adv_video, query + max_p), query - max_p).clamp(0, 255)
                    # 对对抗视频进行特征提取
                    adv_fea = self.t_net(adv_test)

                    # 这行代码的作用是使用对抗样本 adv_fea 作为查询，在目标数据库 self.t_db 中找到最接近目标标签 self.target_label 的样本，返回其索引
                    _, _, adv_retindx = return_label(self.t_db, self.t_dl, adv_fea, self.target_label, self.kappa)

                    # 返回对抗样本特征在在目标数据库 self.t_db 中找到最接近目标标签 self.target_label 的样本的精度
                    Preo2t = return_indx(self.t_db, self.t_dl, adv_fea.detach(), self.target_label, self.tar_retindx,
                                         self.kappa)
                    # 返回对抗样本特征在在目标数据库 self.t_db 中找到最接近目标标签 self.query_label的样本的精度
                    Preo2o = return_indx(self.t_db, self.t_dl, adv_fea.detach(), self.query_label, self.que_retindx,
                                         self.kappa)
                    Preo2tList.append(Preo2t)
                    Preo2oList.append(Preo2o)

                    # 返回对抗样本特征在在目标数据库 self.t_db 中找到最接近目标标签 self.target_label的样本的平均精度
                    advMAPt2t, _, _ = return_label(self.t_db, self.t_dl, adv_fea.detach(), self.target_label,
                                                   self.kappa)

                    advMapo2tlist.append(advMAPt2t)

        # printf(f'Iteration {cnt}: '
        #        f'queries = {int(queries.sum().mean().item())}, '
        #        f'prob = {prev_probs_test:.5f} '
        #        f'prob1 = {prev_probs_test1:.5f} '
        #        f'prob2 = {prev_probs_test2:.5f} '
        #        f'r_int = {cntSucc.sum().item()} '
        #        f'Probo2o:{Preo2o:.5f} '
        #        f'Probo2t:{Preo2t:.5f} '
        #        f'advMAPot2:{advMAPt2t:.4f}  '
        #        f'sparsity:{torch.sum(torch.abs((adv_video - query)) > 0).item()}'
        #        f'Per:{(torch.sum(torch.abs(adv_video - query)) / 602112).item():.4f}')

        printf(f'Iteration {cnt}: '
               f'advMAPot2:{advMAPt2t:.4f}  '
               f'sparsity:{torch.sum(torch.abs((adv_video - query)) > 0).item()}  '
               f'Per:{(torch.sum(torch.abs(adv_video - query)) / 602112).item():.4f}')

        result = {
            'mask_I': mask_I.detach().cpu().numpy(),
            'mask_F': mask_F.detach().cpu().numpy(),
            'max_queries': self.args.max_queries,
            'log_every': log_every,
            'success': True,
            'probs': probs,
            'probs1': probs1,
            'probs2': probs2,
            'adv_video_test': adv_video.cpu().numpy(),
            'query_video': query.clone().cpu().numpy(),
            'target_video': self.target.clone().cpu().numpy(),
            'perturbation': (adv_video - query).cpu().detach().numpy(),
            'Preo2t_item': Preo2t,
            'Preo2o_item': Preo2o,
            'Probs_item': prev_probs_test,
            'Spa_item': torch.sum(torch.abs((adv_video - query)) > 0).item(),
            'Per_item': (torch.sum(torch.abs(adv_video - query)) / 602112).item(),
            'query_path': self.query_path,
            'target_path': self.target_path,
            'target_label': self.target_label.item(),
            'query_label': self.query_label.item(),
            'Preo2oList': np.array(Preo2oList),
            'Preo2tList': np.array(Preo2tList),
            'advMapo2tlist': np.array(advMapo2tlist),
            'max_p': max_p,
            'kappa': np.array(self.kappa),
            'queries': queries,
            'cntSucc': cntSucc,
            'que_retindx': self.que_retindx,
            'tar_retindx': self.tar_retindx,
            'adv_retindx': adv_retindx,
            'max_queries_iterval': max_queries_iterval,
            'MAPo2o': self.MAPo2o,
            'MAPo2t': self.MAPo2t
        }

        return result

    @staticmethod
    def norm2(x):  # 定义2范数
        x = x.squeeze(dim=0)
        assert len(list(x.size())) == 4  # [c, t, h, w]
        x = x.permute(1, 0, 2, 3)  # [t, c, h, w]   #permute(dims) 将tensor的维度换位。
        norm_vec = torch.sqrt(x.float().pow(np.inf).sum(dim=[1, 2, 3])).view(1, 1, -1, 1, 1)  # [b, c, t, h, w]
        # norm_vec = torch.norm(x.float(), p=np.inf)
        norm_vec += (norm_vec == 0).float() * 1e-8
        # print(norm_vec.squeeze())
        return norm_vec

    def norm_inf(x):  # 定义无穷范数
        x = x.squeeze(dim=0)
        assert len(list(x.size())) == 4  # [c, t, h, w]
        x = x.permute(1, 0, 2, 3)  # [t, c, h, w]   #permute(dims) 将tensor的维度换位。
        norm_vec = torch.sqrt(x.float().pow(np.inf).sum(dim=[1, 2, 3])).view(1, 1, -1, 1, 1)  # [b, c, t, h, w]
        norm_vec += (norm_vec == 0).float() * 1e-8
        # print(norm_vec.squeeze())
        return norm_vec

    def get_retrieval_sim_loss(self, q_ret_indx, t_ret_indx):  # q_ret_indx是查询检索列表  t_ret_indx是目标检索列表
        loss = 0.0
        kappa = self.kappa
        t_ret_indx_list = list(t_ret_indx)

        basis = 0.0
        for basis_i in range(kappa):
            basis += (math.log2(basis_i + 1))

        for idx in range(kappa):
            tmp = q_ret_indx[idx]
            if tmp in t_ret_indx:
                idx_t_ret = t_ret_indx_list.index(q_ret_indx[idx])
                tm1 = float((math.log2(kappa - idx + 1)))
                tm2 = float((math.log2(kappa - idx_t_ret + 1)))
                loss += tm1 * tm2 / float(basis)
        return loss

    def get_retrieval_sim_loss2(self, q_ret_indx, t_ret_indx):
        loss = 0.0
        kappa = self.kappa
        t_ret_indx_list = list(t_ret_indx)

        basis = 0.0
        for basis_i in range(kappa):
            basis += float(math.pow(2, kappa - basis_i) - 1)

        for idx in range(kappa):
            tmp = q_ret_indx[idx]
            if tmp in t_ret_indx:
                idx_t_ret = t_ret_indx_list.index(q_ret_indx[idx])
                tm1 = float((math.pow(2, kappa - idx) - 1))
                tm2 = float((math.pow(2, kappa - idx_t_ret) - 1))
                loss += tm1 * tm2 / float(basis)
        return loss

    def get_retrieval_sim_loss1(self, q_ret_indx, t_ret_indx):
        kappa = self.kappa
        AP = []
        Ns = np.array(range(1, kappa + 1)).astype(np.float32)
        for j in range(kappa):  # j= 10 ,判断 query_result 和target_idx 的前10个有多少索引是相似的
            P = (len(set(q_ret_indx[:j + 1]) & set(t_ret_indx[:j + 1])) / Ns[j])  # 计算当前的 presion
            AP.append(P)
        loss = np.mean(np.array(AP))
        return loss

    # ndcg loss
    def get_probs(self, input, return_loss12=False):
        que_retindx = self.que_retindx  # 请求视频的检索列表
        tar_retindx = self.tar_retindx  # 目标视频的检索列表、

        if transability:
            output = self.s_net(input)  # 在代理模型上面运行输入的视频
            r_map, r_lab, r_index = return_label(self.s_db, self.s_dl, output.detach(), self.target_label,
                                                 kappa=self.kappa)
        else:
            output = self.t_net(input)  # 在目标模型上面运行输入的视频
            r_map, r_lab, r_index = return_label(self.t_db, self.t_dl, output.detach(), self.target_label,
                                                 kappa=self.kappa)

        loss1 = self.get_retrieval_sim_loss(r_index,
                                            que_retindx) + 1e-6  # 计算当前检索列表和query检索列表的损失值 对应到公式2中的对抗样本视频到查询视频的NDCG的相似性
        loss2 = self.get_retrieval_sim_loss(r_index,
                                            tar_retindx) + 1e-6  # 计算当前检索列表和target检索列表的损失值 对应到公式2中的对抗样本视频到目标视频的NDCG的相似性

        # dff_cnt = self.find_diff(r_index,que_retindx)
        # loss1, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.query_label, kappa=self.kappa)
        # loss2, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.target_label,kappa=self.kappa)

        t1 = np.array(r_index)
        t2 = np.array(que_retindx)
        t3 = np.array(tar_retindx)

        loss = loss1 - loss2  # SparseQuery中公式2的计算值
        if return_loss12:
            return loss, loss1, loss2, 1
        else:
            return loss

    def get_probs1(self, input, return_loss12=False):
        que_retindx = self.que_retindx  # 请求视频的检索列表
        tar_retindx = self.tar_retindx  # 目标视频的检索列表

        output = self.s_net(input)  # 在代理模型上面运行输入的视频
        r_map, r_lab, r_index = return_label(self.s_db, self.s_dl, output.detach(), self.target_label, kappa=self.kappa)

        loss1 = self.get_retrieval_sim_loss(r_index,
                                            que_retindx) + 1e-6  # 计算当前检索列表和query检索列表的损失值 对应到公式2中的对抗样本视频到查询视频的NDCG的相似性
        loss2 = self.get_retrieval_sim_loss(r_index,
                                            tar_retindx) + 1e-6  # 计算当前检索列表和target检索列表的损失值 对应到公式2中的对抗样本视频到目标视频的NDCG的相似性

        # dff_cnt = self.find_diff(r_index,que_retindx)
        # loss1, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.query_label, kappa=self.kappa)
        # loss2, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.target_label,kappa=self.kappa)

        t1 = np.array(r_index)
        t2 = np.array(que_retindx)
        t3 = np.array(tar_retindx)

        loss = loss1 - loss2  # SparseQuery中公式2的计算值
        if return_loss12:
            return loss, loss1, loss2, 1
        else:
            return loss

    def expand_vector(self, x, size=112):  # 输入了一个一维向量
        batch_size = x.size(0)
        x = x.view(-1, 3, 16, size, size)  # 首先还原成(1 ,3, 16, 112, 112)
        z = torch.zeros(batch_size, 3, 16, size, size).cuda()
        z[:, :, :size, :size] = x  # 然后赋值给z
        return z  # 返回z

    def clamp(self, delta, clean_imgs):
        MEAN = torch.tensor([[[[0.434]]], [[[0.405]]], [[[0.378]]]]).cuda()
        STD = torch.tensor([[[[0.152]]], [[[0.149]]], [[[0.157]]]]).cuda()
        clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
        clamp_delta = (clamp_imgs / 255 - MEAN) / STD - clean_imgs.data
        return clamp_delta

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def sparse_attack(self, query, return_pri=None):  # query  adv_video 是一样的
        # 初始化

        if group_spa:
            query_slider = torch.tensor(self.cut_slider(query))
            I = torch.ones_like(query_slider, requires_grad=False).cuda()  # i是帧内
            mask_I = self.slider_to_img(I)
            print(mask_I.shape)
        else:
            mask_I = torch.ones_like(query, requires_grad=False).cuda()  # 创建一个与查询具有相同形状的张量 mask_I 并将其所有元素设置为 1
        theta = torch.zeros_like(query, requires_grad=False).cuda()  # 创建一个与查询具有相同形状的张量 theta 并将其所有元素设置为 1
        mask_F = torch.ones_like(query, requires_grad=False).cuda()  ## 创建一个与查询具有相同形状的张量 mask_F 并将其所有元素设置为 1
        query_norm = normalization(query).clone()

        input_md = self.s_net
        input_db = self.s_db
        input_dl = self.s_dl
        input_dp = self.t_dp
        input_fea = self.tar_fea_tnet
        input_lab = self.target_label

        self.adv_fealist_snet = []  # 通过替代模型来获取对抗样本的特征
        for mm in range(self.args.n_q):
            adv_video_snet, _ = self.load_video(input_dp[self.tar_retindx_tnet[mm]])  #
            self.adv_fealist_snet.append(input_md(adv_video_snet).detach())

        # 更新theta、mask_I、更新mask_F  就是更新三个参数
        for kk in range(1):
            # 更新theta
            theta = self.update_Theta(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F,
                                      mask_I)
            # 更新变量I
            mask_I = self.update_Intra(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F,
                                       mask_I)
            # 更新变量F
            mask_F = self.update_Inter(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F,
                                       mask_I)

        # theta  = self.update_Theta(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F, mask_I)  #更新theta
        theta = Variable(theta)

        # 测试转移性用的代码，但是没什么转移性

        # if transability:
        #     input_md_tar = self.t_net
        #     input_db_tar = self.t_db
        #     input_dl_tar = self.t_dl
        #     adv_fea = input_md_tar(query_norm + mask_I * mask_F * theta, input_norm=True)
        #     que_fea = input_md_tar(query_norm, input_norm=True)
        #
        #     # _, _, que_index = return_label(self.t_db, self.t_dl, que_fea.detach(), self.target_label,kappa=self.kappa)
        #     # MAPo2t = self.get_retrieval_sim_loss(que_index, self.tar_retindx_tnet) + 1e-6
        #     # _, _, adv_index = return_label(self.t_db, self.t_dl, adv_fea.detach(), self.target_label,kappa=self.kappa)
        #     # ADVo2t = self.get_retrieval_sim_loss(adv_index, self.tar_retindx_tnet) + 1e-6
        #
        #     ADVo2t = return_indx(input_db_tar, input_dl_tar, adv_fea.detach(), input_lab, self.tar_retindx_tnet, self.kappa)
        #     MAPo2t = return_indx(input_db_tar, input_dl_tar, que_fea.detach(), input_lab, self.tar_retindx_tnet, self.kappa)
        #     trans = MAPo2t / ADVo2t
        #     printf(f'Sparse att MAP and trans-ability:{trans:.4f}')

        # ADVo2tIList  = []
        # ADVo2tFList  = []
        # ADVo2tIFList = []
        # for i in range(10):
        #     indices_spaI = torch.randperm(3 * 112 * 112 * 16)
        #     random_maskI = torch.zeros_like(query.reshape(-1)).reshape(-1)
        #     max_sparsity = int(mask_I.sum().item())
        #     random_maskI[indices_spaI[:max_sparsity]]  = 1
        #     random_maskI = random_maskI.reshape(1,3,16,112,112)
        #     indices_spaF =  torch.randperm(16)
        #     random_maskF =  torch.zeros_like(query)
        #     for mm in range(2):
        #         random_maskF[0,:,indices_spaF[mm]] = 1
        #     adv_fea_randI = input_md_tar(query_norm + random_maskI*mask_F*theta, input_norm=True)
        #     adv_fea_randF = input_md_tar(query_norm + mask_I*random_maskF*theta, input_norm=True)
        #     adv_fea_randIF= input_md_tar(query_norm + random_maskI * random_maskF * theta, input_norm=True)
        #
        #     # _, _, advI_index  = return_label(self.t_db, self.t_dl, adv_fea_randI.detach(), self.target_label,kappa=self.kappa)
        #     # _, _, advF_index  = return_label(self.t_db, self.t_dl, adv_fea_randF.detach(), self.target_label,kappa=self.kappa)
        #     # _, _, advIF_index = return_label(self.t_db, self.t_dl, adv_fea_randIF.detach(), self.target_label,kappa=self.kappa)
        #     # ADVo2tI = self.get_retrieval_sim_loss(advI_index, self.tar_retindx_tnet) + 1e-6
        #     # ADVo2tF = self.get_retrieval_sim_loss(advF_index, self.tar_retindx_tnet) + 1e-6
        #     # ADVo2tIF= self.get_retrieval_sim_loss(advIF_index, self.tar_retindx_tnet) + 1e-6
        #
        #     ADVo2tI = return_indx(input_db_tar, input_dl_tar, adv_fea_randI.detach(), input_lab,
        #                          self.tar_retindx_tnet, self.kappa)
        #     ADVo2tF = return_indx(input_db_tar, input_dl_tar, adv_fea_randF.detach(), input_lab,
        #                           self.tar_retindx_tnet, self.kappa)
        #     ADVo2tIF = return_indx(input_db_tar, input_dl_tar, adv_fea_randIF.detach(), input_lab,
        #                           self.tar_retindx_tnet, self.kappa)
        #     ADVo2tIList.append(ADVo2tI)
        #     ADVo2tFList.append(ADVo2tF)
        #     ADVo2tIFList.append(ADVo2tIF)
        # printf(f'Sparse att MAP  {MAPo2t:.4f} /  {ADVo2t:.4f}  randI {mean(ADVo2tIList):.4f}  randF {mean(ADVo2tFList):.4f}'
        #        f'  randIF {mean(ADVo2tIFList):.4f}' )

        return theta, mask_F, mask_I

    # 使用梯度下降根据替代模型的梯度来更新Theta
    def update_Theta(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        global cur_iter, adv_fea  # 当前的迭代次数和对抗视频的特征
        cur_step = self.args.cur_step  #
        args = self.args
        theta = torch.nn.Parameter(theta, requires_grad=True)  # theta 被包装成了一个参数，要求梯度并可以通过反向传播来更新它。
        optimizer = torch.optim.Adam([theta], lr=args.cur_step)  # 使用Adam来对theta进行优化
        train_epochs = args.maxIter_T  # 训练次数
        epsilon = args.epsilon  # 参数，用于控制对抗样本和原始样本之间的差距,影响对抗攻击的成功率和对抗样本的质量。
        max_p = self.max_p

        # 使theta收敛
        gardlist = []
        for cur_iter in range(train_epochs + 1):
            mask_I.requires_grad = False
            mask_F.requires_grad = False
            theta.requires_grad = True
            temp = mask_I * mask_F * theta  # 构建一个临时扰动
            adv_fea = model(query + temp,
                            input_norm=True)  # 使用带噪声的query图像产生一个对应的特征向量,input_norm=True表示将输入的query和噪声都进行归一化。
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off  # 公式1前的的欧式距离函数
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2  # 公式1后面的求取范数

            # 进行组稀疏，然后添加的约束条件
            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            optimizer.zero_grad()  # 将优化器的梯度缓存清零,防止上次梯度对本次迭代的影响
            loss.backward(retain_graph=True)  # 反向传播更新loss
            optimizer.step()  # 更新模型的参数,最小化损失函数参数
            theta = theta.detach()  #

            # 计算某个参数的梯度
            # for k, v in model.named_parameters():
            #     # print(k)
            #     # print(v)
            #     if k == 'net.module.cls_head.fc_cls.bias':
            #         name = k
            #         grad = v.grad.mean().detach().cpu().numpy()
            #         # print(grad.shape)
            #         gardlist.append(grad)
            #         # print(gardlist)
            #     else:
            #         continue
            # theta.data = theta.data.clamp(-args.epsilon, args.epsilon)
            # theta.data = self.clamp(theta, query).clamp(-epsilon, epsilon)

            # 每迭代50次更新一下学习率
            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor,
                               args.lr_min)  # 表示学习率会在每个epoch结束后进行更新，乘以一个衰减因子，直到达到最小学习率 args.lr_min。
                self.adjust_learning_rate(optimizer, cur_step)  # 调整学习率
        # GardList = np.asarray(gardlist)
        # # print(GardList)
        # # GardList = gardlist.reshape(100,-1)
        # # np_to_csv = pd.DataFrame(data = a.reshape(88,-1))
        # # np_to_csv.to_csv('./np_to_csv.csv')
        # np.savetxt(f'./grad/{name}.csv',GardList, delimiter=',')
        # plt.plot(GardList)
        # plt.ylabel('Grad')
        # plt.xlabel('Epoch')
        # plt.show()
        # print('保存成功')

        # 计算对抗样本的相似性
        # MAPo2t, l11, l22 = return_label(db, dl, adv_fea.detach(), self.target_label, self.kappa)
        # printf(f'{cur_iter}, Theta o2tMAP:{MAPo2t:.5f}')
        return theta

    # 使用梯度下降方法来更新F
    def update_Inter(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        args = self.args
        cur_step = self.args.cur_step
        mask_F = torch.nn.Parameter(mask_F, requires_grad=True)  # 将mask_F包装成一个张量，使其可以被优化器更新
        optimizer = torch.optim.Adam([mask_F], lr=cur_step)  # 使用使用Adam来对mask_F进行优化

        train_epochs = args.maxIter_F  # 最大迭代次数300
        for cur_iter in range(1, train_epochs):
            mask_I.requires_grad = False
            mask_F.requires_grad = True
            theta.requires_grad = False
            temp = mask_I * mask_F * theta  # 指的是一个扰动例子

            adv_fea = model(query + temp,
                            input_norm=True)  # # 使用带噪声的query图像产生一个对应的特征向量,input_norm=True表示将输入的query和噪声都进行归一化。
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2

            # 进行组稀疏，然后添加的约束条件
            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            optimizer.zero_grad()  # # 将优化器的梯度缓存清零,防止上次梯度对本次迭代的影响
            loss.backward(retain_graph=True)  # 反向传播更新loss
            optimizer.step()  # # 更新模型的参数,最小化损失函数参数
            #
            # for item in model.named_parameters():
            #     # print(item[1])
            #     if item[0] == 'module.backbone.fc6.weight':
            #         h = item[1].register_hook(lambda grad: print(grad))
            #         print(item[1].grad)
            #     else:
            #         continue

            # updating learning rate
            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)
                self.adjust_learning_rate(optimizer, cur_step)

        mask_F = Variable(mask_F.detach())
        mask_F = self.sort_to_mask(mask_F)
        return mask_F

    # 更新像素掩码mask_I
    def update_Intra(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        args = self.args

        cur_step = args.cur_step  # 学习率
        cur_rho1 = args.cur_rho1  # 扰动的阈值 控制了对抗样本攻击中的最小绝对误差约束，将其控制在[-cur_rho1, cur_rho1]以内
        cur_rho2 = args.cur_rho2  # 扰动的阈值 控制了对抗样本攻击中的最小绝对误差约束，将其控制在[-cur_rho2, cur_rho2]以内
        cur_rho3 = args.cur_rho3  # 扰动的阈值 控制了对抗样本攻击中的最小绝对误差约束，将其控制在[-cur_rho3, cur_rho3]以内

        y1 = torch.ones_like(theta)
        y2 = torch.ones_like(theta)
        z1 = torch.zeros_like(theta)
        z2 = torch.zeros_like(theta)
        z3 = torch.zeros(1).cuda()
        ones = torch.ones_like(theta)
        cur_iter = 0
        err1 = 10
        err2 = 10
        bestMap = 0
        train_epochs = args.maxIter_I  # 最大训练epoch

        while (cur_iter < train_epochs + 1):
            mask_I.requires_grad = True
            mask_F.requires_grad = False
            theta.requires_grad = False

            #y1和y2是通过软阈值函数对mask_I进行约束后得到的结果，用于满足ADMM算法的约束条件。
            y1 = torch.clamp((mask_I.detach() + z1 / cur_rho1), 0.0, 1.0)  # # 将y1张量限制在0到1之间。

            # y2 是用于投影到[0, 1]范围内的中间变量
            y2 = self.project_shifted_lp_ball(mask_I.detach() + z2 / cur_rho2, 0.5 * torch.ones_like(mask_I))

            # 3.update mask_b
            # cnn_grad_mask_b
            temp = mask_I * mask_F * theta

            # args.lambdaL2 = 0
            # # 使用带噪声的query图像产生一个对应的特征向量,input_norm=True表示将输入的query和噪声都进行归一化。
            adv_fea = model(query + temp, input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off  # 用于权衡不同目标或约束条件之间的重要性。
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2 * 0  # 这个0是不是略微有点离谱了
            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            loss.backward(retain_graph=True)

            cnn_grad_mask_b = mask_I.grad  # 保存mask_I的梯度
            mask_b_item = mask_I.sum()  # 求mask_I中元素代码的总和

            # grad_mask_loss1 = cnn_grad_mask_b * args.trade_off * args.trade_off
            # cur_rho1 是当前的拉格朗日乘子步长

            grad_mask_loss1 = cnn_grad_mask_b * args.trade_off  # 梯度与超参数进行一个乘法,用于计算掩码项的梯度损失。
            grad_mask_loss2 = 2 * mask_I * theta * theta * mask_F * mask_F * args.lambdaL2  # 没太看懂这行代码
            grad_mask_b3 = cur_rho1 * (mask_I - y1)  # 这行代码的功能是计算 grad_mask_b3，是一个用于更新拉格朗日乘子的梯度
            grad_mask_b4 = cur_rho2 * (mask_I - y2)  # 这行代码的功能是计算 grad_mask_b4，是一个用于更新拉格朗日乘子的梯度

            # 这里的代码是计算mask项的Lagrangian乘子对应的梯度，具体来说，grad_mask_b5用来计算拉格朗日乘子项grad_mask_loss5的梯度
            grad_mask_b5 = cur_rho3 * (mask_b_item - args.k) * ones

            grad_mask_b = grad_mask_loss1 + grad_mask_loss2 + (z1 + grad_mask_b3) + (z2 + grad_mask_b4) + (
                    z3 * ones + grad_mask_b5)

            mask_I = mask_I - cur_step * grad_mask_b    # 这行代码的作用是更新mask_I，使得其向着最小化loss的方向前进。
            # printf([cnn_grad_mask_b[0][i].detach().cpu().numpy().sum() for i in range(16)])
            mask_I = mask_I.detach()
            z1 = z1 + cur_rho1 * (mask_I - y1)  # 更新z1，
            z2 = z2 + cur_rho2 * (mask_I - y2)
            z3 = z3 + cur_rho3 * (mask_I.sum() - args.k)  # 设置稀疏度  k = 3*16*112*112
            if cur_iter % args.rho_increase_step == 0:
                # cur_rho1 更新为增大一定系数后的结果，并限制在最大值 args.rho1_max 内。
                cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
                cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
                cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)

            # updating learning rate
            if cur_iter % args.lr_decay_step == 0:
                # cur_step 要么按照 args.lr_decay_factor 衰减，要么保持不变（如果衰减后小于 args.lr_min）。
                cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)

            if cur_iter % args.tick_loss_T == 0 and cur_iter % 1000 == 0:
                with torch.no_grad():   # 禁用梯度
                    test_tmp = (mask_I.detach() > 0.5).float() * mask_F * theta
                    test_adv_fea = model(query + test_tmp)

                    # 计算相似性
                    # MAP_ = compute_MAP(db, dl, test_adv_fea.detach(), tar_label)

            if np.isnan(loss.detach().cpu().numpy()):
                exit(0)
            cur_iter += 1
        mask_I = (mask_I > 0.5).float().detach()    # 将mask_I中大于0.5的元素设为1，小于等于0.5的元素设为0

        # printf(f'{cur_iter},Intra MAP: {MAP_}, l1loss:{l1_loss.item():.06f} l2loss:{l2_loss.item():.06f} '
        #       f'k:{(mask_I.detach() > 0.5).float().sum().item()} '
        #       f'g1:{grad_mask_loss1.sum().item():.06f} g2:{grad_mask_loss2.sum().item():.06f} '
        #       f'g3:{grad_mask_b3.sum().item():.06f} g4:{grad_mask_b4.sum().item():.06f} '
        #       f'g5:{grad_mask_b5.sum().item():.06f}')
        return mask_I

    def bim_attack(self, ori_image, adv_image, n_num=100, max_p=30):
        model = self.s_net
        # 收集数据梯度的元素符号
        perturbed_image = adv_image.clone()

        for i in range(1, n_num + 1):
            perturbed_image.requires_grad = True
            adv_fea = model(perturbed_image)
            loss = self.fea_distance(adv_fea, self.tar_fea_snet) * self.args.trade_off

            model.zero_grad()
            loss.backward(retain_graph=True)

            data_grad = perturbed_image.grad.data
            sign_data_grad = data_grad.sign()

            perturbed_image = ori_image - sign_data_grad * max_p
            perturbed_image = torch.max(torch.min(perturbed_image, ori_image + max_p), ori_image - max_p).clamp(0, 255)

            perturbed_image = perturbed_image.detach()

            if i % 100 == 0:
                MAP_ = compute_MAP(self.s_db, self.s_dl, adv_fea.detach(), self.target_label)
                printf(f'BIM attack {MAP_:.5f}')

        return perturbed_image

    def sort_to_mask(self, mask_F):
        n = self.args.n
        ones = torch.zeros_like(mask_F).cuda()
        # mask_F_sort1, mask_F_index1 = torch.norm(theta, p=2, dim=[2, 3, 4]).sort(dim=1, descending=True)

        mask_F_sort, mask_F_index = torch.norm(mask_F, p=2, dim=[1, 3, 4]).sort(dim=1, descending=True)
        for i_batch in range(mask_F_index.shape[0]):
            for j_length in range(n):
                ones[i_batch, :, mask_F_index[i_batch][j_length]] = 1
        return ones

    @staticmethod
    def target_adv_loss(adv_fea, tar_fea):
        k = tar_fea.shape[-1]
        return -(torch.mul(adv_fea, tar_fea).sum()) / k
        # return torch.nn.MSELoss(adv_fea,tar_fea)
        # return -torch.norm(adv_fea - tar_fea, p=2) ** 2

    def fea_distance(self, adv_fea, tar_fea):
        loss = 0
        k = tar_fea.shape[-1]
        for mm in range(self.args.n_q):
            list_adv = self.adv_fealist_snet[mm]
            loss += -(torch.mul(adv_fea, list_adv).sum()) * 1e1 / k
        return loss

        # k = tar_fea.shape[-1]
        # return -(torch.mul(adv_fea, tar_fea).sum())*1e2 / k

        # return self.MSE_loss(adv_fea, tar_fea)
        # return torch.nn.MSELoss(adv_fea, tar_fea)
        # return torch.norm(adv_fea - tar_fea, p=2) ** 2

    def cut_slider(self, query):
        args = self.args
        image_size = self.args.imgsize
        filterSize = self.args.filterSize
        stride = self.args.stride
        num_channels = 1
        batch_size = 1
        query_frame = 16
        stride_frames = 2  # 确定没几个帧作为一个块
        group_size = int(image_size / stride)  # 单边分为多少块
        num_f = 0  # 第几帧被扰动
        num_b = 30  # 第几个块被扰动

        k = int(num_b / group_size)
        n = num_b % group_size

        # 确定步长和滑块大小
        # filterSize = 13  # 指的是步长，每个滑块之间的距离
        # stride = 13  # 步伐，指的是mask滑块的大小

        F = int(np.floor((query_frame - stride_frames) / stride_frames)) + 1  # 计算一个视频的16帧要切成多少块
        P = np.floor((image_size - filterSize) / stride) + 1  # 计算一张图片要走多少个步长
        P = P.astype(np.int32)
        Q = P
        # 一个三维numpy,用做存储切块信息
        index1 = np.ones([query_frame, P * Q, filterSize * filterSize * num_channels], dtype=int)
        index = np.ones([P * Q, filterSize * filterSize * num_channels], dtype=int)
        index_channels = np.ones([1, 3, query_frame, P * Q, filterSize * filterSize * num_channels], dtype=int)

        img_index = np.ones([16, 112, 112], dtype=int)

        tmpidx = 0
        for q in range(Q):
            # plus = 0
            plus1 = q * stride * image_size * num_channels
            for p in range(P):
                index_ = np.array([], dtype=int)
                for i in range(filterSize):
                    index_ = np.append(index_,
                                       np.arange(
                                           p * stride * num_channels + i * image_size * num_channels + plus1,
                                           p * stride * num_channels + i * image_size * num_channels + plus1 + filterSize * num_channels,
                                           dtype=int))
                tmpidx += 1
        for i in range(16):
            index1[i] = index
        for i in range(3):
            index_channels[0][i] = index1
        index1 = np.tile(index1, (batch_size, 1, 1))
        index1[num_f][num_b] = 255
        # for i in range((k - 1) * stride, k * stride):
        #     for j in range((n - 1) * stride, n * stride):
        #         img_index[num_f][i][j] = 255
        return index_channels

    def slider_to_img(self, I):

        image_size = self.args.imgsize
        filterSize = self.args.filterSize
        stride = self.args.stride
        group_size = int(image_size / stride)
        num_slider = int(image_size / stride)

        mask_I = np.ones([1, 3, 16, 112, 112], dtype=int)

        for i in range(3):
            for j in range(16):
                for m in range(num_slider * num_slider):
                    for n in range(stride * stride):
                        mask_I[0][i][j][int(m / num_slider) * 2 + int(n / stride)][stride * (m % num_slider) + n % stride]
        return mask_I

    @staticmethod
    def project_shifted_lp_ball(x, shift_vec):
        shift_x = x - shift_vec
        # compute L2 norm: sum(abs(v)^2)^(1/2)
        norm2_shift = torch.norm(shift_x, p=2)  # 计算shift_x的2-范数。
        n = float(x.numel())  # 返回张量中元素的总数量。
        xp = (n ** (1 / 2) / 2) * (shift_x / norm2_shift) + shift_vec
        return xp

    @staticmethod
    def errorCoef(I, Y):
        Z = I.detach().cpu()
        C = Y.detach().cpu()
        err = torch.sum(torch.abs(Z - C) / Z.numel())
        return err
