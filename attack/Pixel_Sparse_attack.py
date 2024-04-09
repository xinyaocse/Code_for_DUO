"""
The codes are from DUO's work
Implementation of Hybrid Black-box Sparse Adversarial Example Attack on Video Retrieval System
"""

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


class pixel_Sparse_attack(TrainBase):
    def __init__(self, s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device, classes):
        super(pixel_Sparse_attack, self).__init__(s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device,
                                         classes)
        pass

    def attack(self):
        global que_tarlabel_tnet, i
        cnt = 0
        self.APo2o = []
        self.APo2t = []
        self.t_mapMean = []
        self.spaMean = []
        datalen = self.query_iter.shape[0] if self.query_iter.shape[0] < 6 else 5
        for i_batch in range(datalen):
            i_batch_data = self.query_iter[i_batch]

            self.query_label = torch.tensor([int(i_batch_data[1])])
            # print(self.query_label)
            self.query_path = i_batch_data[2][0][:-15]
            self.query_path_list = i_batch_data[2]

            self.target_label = torch.tensor([int(i_batch_data[3])])
            self.target_path = i_batch_data[4][0][:-15]
            self.target_path_list = i_batch_data[4]

            self.query, _ = self.load_video(self.query_path_list)
            self.target, _ = self.load_video(self.target_path_list)

            MAPo2o, que_retlabel_tnet, que_retindx_tnet = return_label(self.t_db,
                                                                       self.t_dl,
                                                                       self.t_net(self.query),
                                                                       self.query_label,
                                                                       self.kappa)


            MAPo2t, tar_retlabel_tnet, self.tar_retindx_tnet \
                = return_label(self.t_db,
                               self.t_dl,
                               self.t_net(self.target),
                               self.query_label,
                               self.kappa)


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

        self.tar_fea_snet = self.s_net(self.target).detach()
        self.tar_fea_tnet = self.t_net(self.target).detach()  #
        self.que_fea_tnet = self.t_net(query).detach()

        MAPt2t_tnet, m1, self.tar_retindx = return_label(self.t_db,
                                                         self.t_dl,
                                                         self.tar_fea_tnet,
                                                         self.target_label,
                                                         self.kappa)
        MAPo2o_tnet, m2, self.que_retindx = return_label(self.t_db,
                                                         self.t_dl,
                                                         self.que_fea_tnet,
                                                         self.query_label,
                                                         self.kappa)
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
            for rr in range(iteration_num):
                path = f'iterval_{str(int(self.args.interval))}_' \
                       f'n_{str(int(self.args.n))}_' \
                       f'k_{str(int(self.args.k / 1e4))}_' \
                       f'maxp_{str(int(self.max_p))}_' \
                       f'kappa_{str(int(self.kappa))}_' \
                       f'ori_{self.query_path.split("/")[-2]}_' \
                       f'tar_{self.target_path.split("/")[-2]}_{str(rr)}'

                result = self.HSA_SIMBA(query)
                self.save_dic(result, path)

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

    def heuristic_attack(self, spa_ratio, key_frame_num):
        # spa_ratio   = 0.2
        from attack.ATT_FUNC import SpectralResidual, get_cv2_func
        cv2_func = get_cv2_func(0)
        spatial_mask = []
        for i in range(16):
            this_mask = SpectralResidual(cv2_func, self.query.cpu()[0, :, i], spa_ratio)
            spatial_mask.append(this_mask)
        spatial_mask = torch.from_numpy(np.array(spatial_mask)).unsqueeze(0).permute(0, 4, 1, 2,
                                                                                     3).cuda()
        advNoise = (self.target - self.query)
        all_frames = [i for i in range(16)]
        query = 0
        score_dict = {}
        tar_probs, _, _, _ = self.get_probs(self.target, return_loss12=True)
        que_probs, _, _, _ = self.get_probs(self.query, return_loss12=True)

        for i in all_frames:
            tmp_frames = [_ for _ in all_frames if _ != i]
            MASK = torch.zeros_like(self.query).cuda()
            MASK[0, :, tmp_frames, :, :] = 1
            MASK = MASK * spatial_mask
            adv_video = advNoise * MASK + self.query
            adv_probs, _, _, _ = self.get_probs(adv_video, return_loss12=True)
            score_dict[i] = adv_probs
        sorted_items = sorted(score_dict.items(), key=lambda x: -x[1], reverse=True)
        key_frame_indx = [sorted_items[i][0] for i in range(key_frame_num)]
        MASK = torch.zeros_like(self.query).cuda()
        MASK[0, :, key_frame_indx, :, :] = 1
        MASK = MASK * spatial_mask
        return MASK

    def HSA_SIMBA(self, query):
        global trans, group_spa, transability, indices, prev_probs_test, left_probs, right_probs, theta_test, mask_I, mask_F, expand_dims, adv_retindx, max_queries, theta_right, adv_right, advMAPt2t, Preo2o, Preo2t, interval, max_queries_iterval
        max_iters = int(1e5)  #
        log_every = 10  #
        pixel_attack = True  #
        transability = False  #
        group_spa = False  #
        query_size = query.size(4)  #
        query_frame = query.size(2)  #
        batch_size = query.size(0)  # batch_size  = 1
        dims_intra = query_size  #
        dims_inter = query_frame
        order = 'rand'


        if order == 'rand':
            expand_dims = dims_intra
        n_dims = 3 * dims_inter * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)

        probs1 = np.zeros((max_iters))
        probs2 = np.zeros((max_iters))
        probs = np.zeros((max_iters))
        queries = np.zeros((max_iters))  #
        cntSucc = np.zeros((max_iters))
        advMapo2tlist = []
        Preo2tList = []
        Preo2oList = []
        max_p = self.max_p


        if transability:
            adv_video = query.clone()
            theta, mask_F, mask_I = self.sparse_attack(adv_video)
            prev_probs_test, prev_probs_test1, prev_probs_test2, _ = self.get_probs(adv_video, return_loss12=True)
        else:
            adv_video = query.clone()
            theta, mask_F, mask_I = self.sparse_attack(adv_video)

            prev_probs_test, prev_probs_test1, prev_probs_test2, _ = self.get_probs(adv_video, return_loss12=True)

        if pixel_attack:
            trans = lambda z: z


        left_probs = 10  #
        right_probs = 10
        mask_F = torch.ones_like(query)
        mask_I = torch.ones_like(query)

        max_queries = int(self.args.max_queries)
        interval = int(self.args.interval)

        if self.args.attack_method == 'HSA_SIMBA_RAND':
            # 可替换为查找文件确定 k。
            printf(f'Running Rand Select Mec...........')
            max_queries_iterval = int(max_queries / interval)
            indices = torch.randperm(
                3 * dims_inter * dims_intra * dims_intra)
            printf(f'Fininsh Random attack max_queries:{str(int(max_queries))} ...........')

        if self.args.attack_method == 'HSA_SIMBA_HEURISTIC':
            printf(f'Running Heuristic indics Select Mec  ...........')
            spa_ratio, key_frame = float((self.args.k / 602112)), self.args.n

            mask_H = self.heuristic_attack(spa_ratio, key_frame)
            return_data, order = torch.sort(torch.abs((mask_H).reshape(-1)), descending=True)
            indices = order.clone()
            max_queries_iterval = int(torch.sum(torch.abs(mask_H)).item())
            printf(f'Fininsh Heuristic indics Select Mec with k:{str(int(self.args.k))} n:{str(int(self.args.n))}'
                   f' SPA:{str(max_queries_iterval)}...........')

        cnt = 0
        for mm in range(int(interval)):
            if self.args.attack_method == 'HSA_SIMBA_SPA':
                printf(f'Running Spare Mec on {str(int(mm))}...........')
                theta, mask_F, mask_I = self.sparse_attack(adv_video)




                return_data, order = torch.sort(torch.abs((mask_F * mask_I * theta).reshape(-1)), descending=True)
                indices = order.clone()
                max_queries_iterval = int(torch.sum(mask_F * mask_I).item())

                printf(f'Fininsh Spare Mec with k:{str(int(self.args.k))} n:{str(int(self.args.n))}'
                       f' SPA:{str(int(torch.sum(torch.abs(mask_F * mask_I)).item()))}...........')

            for k in range(max_queries_iterval):
                cnt += 1

                dim = indices[k]
                diff = torch.zeros(1, n_dims)  # 1 * 3 * 16 * 112 * 112
                diff[:, dim] = max_p
                left_vec = x + diff
                right_vec = x - diff

                # try negative directions

                theta_left = trans(self.expand_vector(left_vec, expand_dims))  # 将向量还原成视频
                adv_left = torch.max(torch.min(theta_left + adv_video, query + max_p), query - max_p).clamp(0, 255)

                left_probs, left_probs1, left_probs2, _ = self.get_probs(adv_left, return_loss12=True)

                queries_k = 1
                if left_probs < prev_probs_test:
                    x[0, dim] = max_p  #
                    prev_probs_test = left_probs
                    cntSucc[cnt] = 1  #
                    adv_video = adv_left.clone()
                else:
                    # try positive directions
                    theta_right = trans(self.expand_vector(right_vec, expand_dims)).clone()
                    queries_k = 2


                    adv_right = torch.max(torch.min(theta_left + adv_video, query + max_p), query - max_p).clamp(0, 255)
                    right_probs, right_probs1, right_probs2, _ = self.get_probs(adv_right, return_loss12=True)

                if right_probs < prev_probs_test:
                    adv_right = torch.max(torch.min(theta_right + adv_video, query + max_p), query - max_p).clamp(0,
                                                                                                                  255)
                    x[0, dim] = -max_p
                    prev_probs_test = right_probs
                    adv_video = adv_right.clone()
                    cntSucc[cnt] = 1

                probs2[cnt] = prev_probs_test2
                probs1[cnt] = prev_probs_test1
                probs[cnt] = prev_probs_test
                queries[cnt] = queries_k  #

                # if int(torch.sum(torch.abs((adv_video - query)) > 0).item()) > max_SPA:
                #     break
                # if sum(queries) > max_queries:
                #     break

                if k % 100 == 0:
                    theta_test = trans(self.expand_vector(x, expand_dims)).clone()
                    adv_test = torch.max(torch.min(theta_test + adv_video, query + max_p), query - max_p).clamp(0, 255)
                    adv_fea = self.t_net(adv_test)

                    _, _, adv_retindx = return_label(self.t_db, self.t_dl, adv_fea, self.target_label, self.kappa)

                    Preo2t = return_indx(self.t_db, self.t_dl, adv_fea.detach(), self.target_label, self.tar_retindx,
                                         self.kappa)
                    Preo2o = return_indx(self.t_db, self.t_dl, adv_fea.detach(), self.query_label, self.que_retindx,
                                         self.kappa)
                    Preo2tList.append(Preo2t)
                    Preo2oList.append(Preo2o)

                    advMAPt2t, _, _ = return_label(self.t_db, self.t_dl, adv_fea.detach(), self.target_label,
                                                   self.kappa)

                    advMapo2tlist.append(advMAPt2t)



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
        x = x.permute(1, 0, 2, 3)  # [t, c, h, w]
        norm_vec = torch.sqrt(x.float().pow(np.inf).sum(dim=[1, 2, 3])).view(1, 1, -1, 1, 1)  # [b, c, t, h, w]
        # norm_vec = torch.norm(x.float(), p=np.inf)
        norm_vec += (norm_vec == 0).float() * 1e-8
        # print(norm_vec.squeeze())
        return norm_vec

    def norm_inf(x):
        x = x.squeeze(dim=0)
        assert len(list(x.size())) == 4  # [c, t, h, w]
        x = x.permute(1, 0, 2, 3)
        norm_vec = torch.sqrt(x.float().pow(np.inf).sum(dim=[1, 2, 3])).view(1, 1, -1, 1, 1)  # [b, c, t, h, w]
        norm_vec += (norm_vec == 0).float() * 1e-8
        # print(norm_vec.squeeze())
        return norm_vec

    def get_retrieval_sim_loss(self, q_ret_indx, t_ret_indx):
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
        for j in range(kappa):  #
            P = (len(set(q_ret_indx[:j + 1]) & set(t_ret_indx[:j + 1])) / Ns[j])
            AP.append(P)
        loss = np.mean(np.array(AP))
        return loss

    # ndcg loss
    def get_probs(self, input, return_loss12=False):
        que_retindx = self.que_retindx
        tar_retindx = self.tar_retindx

        if transability:
            output = self.s_net(input)
            r_map, r_lab, r_index = return_label(self.s_db, self.s_dl, output.detach(), self.target_label,
                                                 kappa=self.kappa)
        else:
            output = self.t_net(input)
            r_map, r_lab, r_index = return_label(self.t_db, self.t_dl, output.detach(), self.target_label,
                                                 kappa=self.kappa)

        loss1 = self.get_retrieval_sim_loss(r_index,
                                            que_retindx) + 1e-6
        loss2 = self.get_retrieval_sim_loss(r_index,
                                            tar_retindx) + 1e-6


        t1 = np.array(r_index)
        t2 = np.array(que_retindx)
        t3 = np.array(tar_retindx)

        loss = loss1 - loss2
        if return_loss12:
            return loss, loss1, loss2, 1
        else:
            return loss

    def get_probs1(self, input, return_loss12=False):
        que_retindx = self.que_retindx
        tar_retindx = self.tar_retindx

        output = self.s_net(input)
        r_map, r_lab, r_index = return_label(self.s_db, self.s_dl, output.detach(), self.target_label, kappa=self.kappa)

        loss1 = self.get_retrieval_sim_loss(r_index,
                                            que_retindx) + 1e-6
        loss2 = self.get_retrieval_sim_loss(r_index,
                                            tar_retindx) + 1e-6


        t1 = np.array(r_index)
        t2 = np.array(que_retindx)
        t3 = np.array(tar_retindx)

        loss = loss1 - loss2
        if return_loss12:
            return loss, loss1, loss2, 1
        else:
            return loss

    def expand_vector(self, x, size=112):
        batch_size = x.size(0)
        x = x.view(-1, 3, 16, size, size)
        z = torch.zeros(batch_size, 3, 16, size, size).cuda()
        z[:, :, :size, :size] = x
        return z

    def clamp(self, delta, clean_imgs):
        MEAN = torch.tensor([[[[0.434]]], [[[0.405]]], [[[0.378]]]]).cuda()
        STD = torch.tensor([[[[0.152]]], [[[0.149]]], [[[0.157]]]]).cuda()
        clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
        clamp_delta = (clamp_imgs / 255 - MEAN) / STD - clean_imgs.data
        return clamp_delta

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def sparse_attack(self, query, return_pri=None):


        if group_spa:
            query_slider = torch.tensor(self.cut_slider(query))
            I = torch.ones_like(query_slider, requires_grad=False).cuda()
            mask_I = self.slider_to_img(I)
            print(mask_I.shape)
        else:
            mask_I = torch.ones_like(query, requires_grad=False).cuda()
        theta = torch.zeros_like(query, requires_grad=False).cuda()  #
        mask_F = torch.ones_like(query, requires_grad=False).cuda()
        query_norm = normalization(query).clone()

        input_md = self.s_net
        input_db = self.s_db
        input_dl = self.s_dl
        input_dp = self.t_dp
        input_fea = self.tar_fea_tnet
        input_lab = self.target_label

        self.adv_fealist_snet = []
        for mm in range(self.args.n_q):
            adv_video_snet, _ = self.load_video(input_dp[self.tar_retindx_tnet[mm]])  #
            self.adv_fealist_snet.append(input_md(adv_video_snet).detach())

        # Updating theta, mask_I, and updating mask_F is to update the three parameters
        for kk in range(1):
            # Updating theta
            theta = self.update_Theta(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F,
                                      mask_I)
            # Updating I
            mask_I = self.update_Intra(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F,
                                       mask_I)
            # Updating F
            mask_F = self.update_Inter(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta, mask_F,
                                       mask_I)


        theta = Variable(theta)



        return theta, mask_F, mask_I

    def update_Theta(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        global cur_iter, adv_fea
        cur_step = self.args.cur_step  #
        args = self.args
        theta = torch.nn.Parameter(theta, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=args.cur_step)
        train_epochs = args.maxIter_T
        epsilon = args.epsilon
        max_p = self.max_p

        # 使theta收敛
        gardlist = []
        for cur_iter in range(train_epochs + 1):
            mask_I.requires_grad = False
            mask_F.requires_grad = False
            theta.requires_grad = True
            temp = mask_I * mask_F * theta
            adv_fea = model(query + temp,
                            input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2



            loss = l1_loss + l2_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)  #
            optimizer.step()
            theta = theta.detach()  #


            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor,
                               args.lr_min)
                self.adjust_learning_rate(optimizer, cur_step)

        return theta


    def update_Inter(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        args = self.args
        cur_step = self.args.cur_step
        mask_F = torch.nn.Parameter(mask_F, requires_grad=True)
        optimizer = torch.optim.Adam([mask_F], lr=cur_step)

        train_epochs = args.maxIter_F
        for cur_iter in range(1, train_epochs):
            mask_I.requires_grad = False
            mask_F.requires_grad = True
            theta.requires_grad = False
            temp = mask_I * mask_F * theta

            adv_fea = model(query + temp,
                            input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2


            loss = l1_loss + l2_loss

            optimizer.zero_grad()  #
            loss.backward(retain_graph=True)  #
            optimizer.step()
            #

            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)
                self.adjust_learning_rate(optimizer, cur_step)

        mask_F = Variable(mask_F.detach())
        mask_F = self.sort_to_mask(mask_F)
        return mask_F

    # 更新像素掩码mask_I
    def update_Intra(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        args = self.args

        cur_step = args.cur_step
        cur_rho1 = args.cur_rho1
        cur_rho2 = args.cur_rho2  #
        cur_rho3 = args.cur_rho3  #

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
        train_epochs = args.maxIter_I

        while (cur_iter < train_epochs + 1):
            mask_I.requires_grad = True
            mask_F.requires_grad = False
            theta.requires_grad = False


            y1 = torch.clamp((mask_I.detach() + z1 / cur_rho1), 0.0, 1.0)

            y2 = self.project_shifted_lp_ball(mask_I.detach() + z2 / cur_rho2, 0.5 * torch.ones_like(mask_I))

            # 3.update mask_b
            # cnn_grad_mask_b
            temp = mask_I * mask_F * theta



            adv_fea = model(query + temp, input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2 * 0
            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            loss.backward(retain_graph=True)

            cnn_grad_mask_b = mask_I.grad
            mask_b_item = mask_I.sum()


            grad_mask_loss1 = cnn_grad_mask_b * args.trade_off  #
            grad_mask_loss2 = 2 * mask_I * theta * theta * mask_F * mask_F * args.lambdaL2  #
            grad_mask_b3 = cur_rho1 * (mask_I - y1)
            grad_mask_b4 = cur_rho2 * (mask_I - y2)

            grad_mask_b5 = cur_rho3 * (mask_b_item - args.k) * ones

            grad_mask_b = grad_mask_loss1 + grad_mask_loss2 + (z1 + grad_mask_b3) + (z2 + grad_mask_b4) + (
                    z3 * ones + grad_mask_b5)

            mask_I = mask_I - cur_step * grad_mask_b
            # printf([cnn_grad_mask_b[0][i].detach().cpu().numpy().sum() for i in range(16)])
            mask_I = mask_I.detach()
            z1 = z1 + cur_rho1 * (mask_I - y1)
            z2 = z2 + cur_rho2 * (mask_I - y2)
            z3 = z3 + cur_rho3 * (mask_I.sum() - args.k)
            if cur_iter % args.rho_increase_step == 0:
                cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
                cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
                cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)

            # updating learning rate
            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)

            if cur_iter % args.tick_loss_T == 0 and cur_iter % 1000 == 0:
                with torch.no_grad():   # 禁用梯度
                    test_tmp = (mask_I.detach() > 0.5).float() * mask_F * theta
                    test_adv_fea = model(query + test_tmp)


            if np.isnan(loss.detach().cpu().numpy()):
                exit(0)
            cur_iter += 1
        mask_I = (mask_I > 0.5).float().detach()    #


        return mask_I

    def bim_attack(self, ori_image, adv_image, n_num=100, max_p=30):
        model = self.s_net

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
        norm2_shift = torch.norm(shift_x, p=2)  #
        n = float(x.numel())
        xp = (n ** (1 / 2) / 2) * (shift_x / norm2_shift) + shift_vec
        return xp

    @staticmethod
    def errorCoef(I, Y):
        Z = I.detach().cpu()
        C = Y.detach().cpu()
        err = torch.sum(torch.abs(Z - C) / Z.numel())
        return err
