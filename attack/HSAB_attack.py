"""
The codes are from Enlang Li's work
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
import cv2
from utils.argparser import *
from numpy import *
import math
from attack.base import *


class HSAB_attack(TrainBase):
    def __init__(self, s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device, classes):
        super(HSAB_attack, self).__init__(s_net, t_net, query_iter, target_iter, s_db, s_dl, t_db, t_dl, parser, device,
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
        # datalen = 2
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
        global path
        current_time = time.time()

        global result, adv_imgs

        self.tar_fea_snet = self.s_net(self.target).detach()
        self.tar_fea_tnet = self.t_net(self.target).detach()
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
            s1 = '_'.join(self.query_path.split("/")[-2].split("_")[0:6])
            s2 = '_'.join(self.target_path.split("/")[-2].split("_")[0:6])
            s1 = s1 if len(s1) < 30 else s1[0:31]
            s2 = s2 if len(s2) < 30 else s2[0:31]
            for rr in range(iteration_num):
                path_mat = f'iterval_{str(int(self.args.interval))}_' \
                           f'n_{str(int(self.args.n))}_' \
                           f'k_{str(int(self.args.k / 1e4))}_' \
                           f'maxp_{str(int(self.max_p))}_' \
                           f'kappa_{str(int(self.kappa))}_' \
                           f'ori_{s1}_' \
                           f'tar_{s2}_{str(rr)}'

                result = self.HSA_SIMBA(query)

                self.save_dic(result, path_mat)

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

    # Corresponds to the Vanilla attack in the paper
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
        global indic_m, trans, transability, indices, prev_probs_test, left_probs, right_probs, theta_test, mask_I, mask_F, expand_dims, adv_retindx, max_queries, theta_right, adv_right, advMAPt2t, Preo2o, Preo2t, interval, max_queries_iterval
        max_iters = int(1e5)
        indic_m = []
        log_every = 10
        pixel_attack = True
        transability = self.args.trans
        query_size = query.size(4)
        query_frame = query.size(2)
        batch_size = query.size(0)
        dims_intra = query_size
        dims_inter = query_frame

        image_size = query_size
        maskSize = self.args.mask_size
        channels = self.args.channels
        stride = self.args.stride

        P = np.floor((image_size - maskSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P


        for i in range(P * Q):
            for j in range(maskSize * maskSize):
                ls = []
                m = int(maskSize * int((i // P)) + int(j // maskSize))
                n = int(int((i % P)) * maskSize + int(j % maskSize))
                ls = [i, j, m, n]
                indic_m.append(ls)

        order = 'rand'


        if order == 'rand':
            expand_dims = dims_intra
        n_dims = 3 * dims_inter * expand_dims * expand_dims
        n_gdims = channels * query_frame * P * Q
        x1 = torch.zeros(batch_size, n_dims)
        x = torch.zeros_like(query).cuda()

        probs1 = np.zeros((max_iters))
        probs2 = np.zeros((max_iters))
        probs = np.zeros((max_iters))
        queries = np.zeros((max_iters))
        cntSucc = np.zeros((max_iters))
        advMapo2tlist = []
        Preo2tList = []
        Preo2oList = []
        Preo2tGPList = []
        max_p = self.max_p


        if pixel_attack:
            trans = lambda z: z

        adv_video = query.clone()
        # theta, mask_F, mask_I = self.sparse_attack(adv_video)
        prev_probs_test, prev_probs_test1, prev_probs_test2, _ = self.get_probs(adv_video,
                                                                                return_loss12=True)

        left_probs = 10
        right_probs = 10
        mask_F = torch.ones_like(query)
        mask_I = torch.ones_like(query)
        theta = torch.ones_like(query)

        max_queries = int(self.args.max_queries)
        interval = int(self.args.interval)
        max_SPA = self.args.max_spa

        if self.args.attack_method == 'HSA_SIMBA_RAND':  # Pick a random spot
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
        # int(interval)
        for mm in range(int(interval)):
            if self.args.attack_method == 'HSA_SIMBA_SPA':  # Sparse algorithm selection
                printf(f'Running Spare Mec on {str(int(mm))}...........')
                theta_G, mask_FG, mask_G = self.sparse_attack(adv_video)
                mask_I = self.G2I(mask_G)
                # print(torch.sum(mask_I))
                mask_F = self.G2I(mask_FG)
                # print(torch.sum(mask_F))



                return_data, order = torch.sort(torch.abs((mask_FG * mask_G * theta_G).reshape(-1)),
                                                descending=True)
                indices = order.clone()

                max_queries_iterval = int(torch.sum(mask_FG * mask_G).item())  #

                printf(f'Fininsh Spare Mec with k:{str(int(self.args.k))} n:{str(int(self.args.n))}'
                       f' SPA:{str(int(torch.sum(torch.abs(mask_FG * mask_G)).item()))}...........')


            query_norm = normalization(query).clone()
            input_md = self.s_net
            input_db = self.s_db
            input_dl = self.s_dl
            input_dp = self.t_dp
            input_fea = self.tar_fea_tnet
            input_lab = self.target_label

            for k in range(max_queries_iterval):
                # print(f'{mm}-{k}')
                cnt += 1

                dim = indices[k]
                diff_vec = torch.zeros(1, n_gdims)  # (1,3*16*56*56)
                diff_vecx = torch.zeros(1, n_gdims)
                diff_vec[:, dim] = max_p
                diff_G = diff_vec.view(-1, channels, query_frame, P * Q)  # (1,3,16,56*56)
                diff = self.G2I(diff_G).cuda()
                diff = diff.cuda()
                # diff_g = diff.view(diff.shape[0], diff.shape[1] * diff.shape[2] * diff.shape[3] * diff.shape[4])
                # x = torch.zeros_like(query).cuda()
                left_vec = x + diff
                right_vec = x - diff

                # try negative directions

                # theta_left = trans(self.expand_vector(left_vec, expand_dims))
                theta_left = left_vec
                adv_left = torch.max(torch.min(theta_left + adv_video, query + max_p), query - max_p).clamp(0, 255)

                left_probs, left_probs1, left_probs2, _ = self.get_probs(adv_left, return_loss12=True)

                queries_k = 1
                if left_probs < prev_probs_test:
                    diff_vecx[0, dim] = max_p  #
                    diff_Gx = diff_vecx.view(-1, channels, query_frame, P * Q)
                    diff_x = self.G2I(diff_Gx)
                    diff_x = diff_x.cuda()
                    x = x + diff_x
                    prev_probs_test = left_probs
                    cntSucc[cnt] = 1
                    adv_video = adv_left.clone()
                else:
                    # try positive directions
                    # theta_right = trans(self.expand_vector(right_vec, expand_dims)).clone()
                    theta_right = right_vec.clone()
                    queries_k = 2


                    adv_right = torch.max(torch.min(theta_left + adv_video, query + max_p), query - max_p).clamp(0, 255)
                    # adv_right = torch.max(torch.min(theta_right + adv_video, query + max_p), query - max_p).clamp(0,255)
                    right_probs, right_probs1, right_probs2, _ = self.get_probs(adv_right, return_loss12=True)

                if right_probs < prev_probs_test:

                    adv_right = torch.max(torch.min(theta_right + adv_video, query + max_p), query - max_p).clamp(0,
                                                                                                                  255)
                    diff_vecx[0, dim] = -max_p
                    diff_Gx = diff_vecx.view(-1, channels, query_frame, P * Q)
                    diff_x = self.G2I(diff_Gx)
                    diff_x = diff_x.cuda()
                    x = x + diff_x
                    prev_probs_test = right_probs
                    adv_video = adv_right.clone()
                    cntSucc[cnt] = 1


                x = x.clamp(-max_p , max_p)
                probs2[cnt] = prev_probs_test2
                probs1[cnt] = prev_probs_test1
                probs[cnt] = prev_probs_test
                queries[cnt] = queries_k

                # if int(torch.sum(torch.abs((adv_video - query)) > 0).item()) > max_SPA or sum(queries) > max_queries:
                #     break
                if int(torch.sum(torch.abs((adv_video - query)) > 0).item()) > max_SPA:
                    break
                # #
                # if sum(queries) > max_queries:
                #     break

                if k % 100 == 0:

                    # theta_test = trans(self.expand_vector(x, expand_dims)).clone()
                    theta_test = x.clone()
                    # theta_test = self.update_ThetaP(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta_test, mask_F, mask_I)

                    adv_test = torch.max(torch.min(theta_test + adv_video, query + max_p), query - max_p).clamp(0, 255)

                    adv_fea = self.t_net(adv_test)


                    _, _, adv_retindx = return_label(self.t_db, self.t_dl, adv_fea, self.target_label, self.kappa)


                    Preo2t = return_indx(self.t_db, self.t_dl, adv_fea.detach(), self.target_label, self.tar_retindx,
                                         self.kappa)

                    Preo2o = return_indx(self.t_db, self.t_dl, adv_fea.detach(), self.query_label, self.que_retindx,
                                         self.kappa)
                    Preo2tList.append(Preo2t)
                    Preo2oList.append(Preo2o)

                    #
                    advMAPt2t, _, _ = return_label(self.t_db, self.t_dl, adv_fea.detach(), self.target_label,
                                                   self.kappa)

                    advMapo2tlist.append(advMAPt2t)


            query_norm = normalization(query).clone()
            input_md = self.s_net
            input_db = self.s_db
            input_dl = self.s_dl
            input_dp = self.t_dp
            input_fea = self.tar_fea_tnet
            input_lab = self.target_label

            perturbetion = x
            mask_IG1 = abs(torch.div(perturbetion.int(), int(30)))


            theta_P = torch.zeros_like(query, requires_grad=False).cuda()
            theta_P = self.update_ThetaP(query_norm, input_md, input_db, input_dl, input_fea, input_lab, x, mask_F, mask_IG1)
            adv_v1 = query + theta_P
            adv_v = torch.clamp(adv_v1, 0, 255)
            adv_f = self.t_net(adv_v)
            Preo2t_P = return_indx(self.t_db, self.t_dl, adv_f.detach(), self.target_label, self.tar_retindx,
                                   self.kappa)
            Preo2tGPList.append(Preo2t_P)



        printf(f'Iteration {cnt}: '
               f'advMAPot2:{advMAPt2t:.4f}  '
               f'sparsity:{torch.sum(torch.abs((adv_video - query)) > 0).item()}  '
               f'Per:{(torch.sum(torch.abs(adv_video - query)) / 602112).item():.4f}  '
               f'queries {int(queries.sum().mean().item())} ')

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
            'MAPo2t': self.MAPo2t,
            'Mapo2t_GP': Preo2tGPList
        }

        return result

    @staticmethod
    def norm2(x):
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
        x = x.permute(1, 0, 2, 3)  # [t, c, h, w]
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
        for j in range(kappa):
            P = (len(set(q_ret_indx[:j + 1]) & set(t_ret_indx[:j + 1])) / Ns[j])
            AP.append(P)
        loss = np.mean(np.array(AP))
        return loss

    # ndcg loss
    def get_probs(self, input, return_loss12=False):
        que_retindx = self.que_retindx
        tar_retindx = self.tar_retindx

        output = self.t_net(input)  #
        r_map, r_lab, r_index = return_label(self.t_db, self.t_dl, output.detach(), self.target_label,
                                             kappa=self.kappa)
        #
        if transability:
            output = self.s_net(input)
            r_map, r_lab, r_index = return_label(self.s_db, self.s_dl, output.detach(), self.target_label,
                                                 kappa=self.kappa)
        else:
            output = self.t_net(input)
            r_map, r_lab, r_index = return_label(self.t_db, self.t_dl, output.detach(), self.target_label,
                                                 kappa=self.kappa)
        #
        loss1 = self.get_retrieval_sim_loss(r_index,
                                            que_retindx) + 1e-6
        loss2 = self.get_retrieval_sim_loss(r_index,
                                            tar_retindx) + 1e-6

        # # dff_cnt = self.find_diff(r_index,que_retindx)
        # loss1, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.query_label, kappa=self.kappa)
        # loss2, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.target_label,kappa=self.kappa)

        # t1 = np.array(r_index)
        # t2 = np.array(que_retindx)
        # t3 = np.array(tar_retindx)

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

        # dff_cnt = self.find_diff(r_index,que_retindx)
        # loss1, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.query_label, kappa=self.kappa)
        # loss2, _, _ = return_label(self.t_db, self.t_dl, self.t_net(input).detach(), self.target_label,kappa=self.kappa)

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
        z[:, :, :size, :size] = x  #
        return z

    def expand_G(self, G, P):

        G = G.view(-1, 16, P * P)

        return G

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

        image_size = self.args.imgsize
        maskSize = self.args.mask_size
        channels = self.args.channels
        query_frame = self.args.num_frames
        # filterSize = stride = maskSize
        stride = self.args.stride
        batch_size = 1

        # Calculate p and q
        P = np.floor((image_size - maskSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P

        mask_I = torch.ones_like(query, requires_grad=False).cuda()
        theta = torch.zeros_like(query, requires_grad=False).cuda()
        mask_F = torch.ones_like(query, requires_grad=False).cuda()
        index_G = torch.ones_like(torch.from_numpy(np.ones([batch_size, channels, query_frame, P * Q], dtype=float)))

        mask_G = torch.ones_like(index_G, requires_grad=False).cuda()
        theta_G = torch.zeros_like(index_G, requires_grad=False).cuda()
        mask_FG = torch.ones_like(index_G, requires_grad=False).cuda()
        mask_FG1 = torch.zeros_like(index_G, requires_grad=False).cuda()
        mask_IG1 = torch.zeros_like(index_G, requires_grad=False).cuda()

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

            theta_G = self.update_Theta(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta_G,
                                        mask_FG, mask_G)


            mask_G = self.update_Intra(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta_G,
                                       mask_FG, mask_G)


            mask_FG = self.update_Inter(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta_G,
                                        mask_FG, mask_G)


        theta_G = self.update_Theta(query_norm, input_md, input_db, input_dl, input_fea, input_lab, theta_G,
                                    mask_FG, mask_G)
        theta_G = Variable(theta_G)



        return theta_G, mask_FG, mask_G


    def update_Theta(self, query, model, db, dl, tar_fea, tar_label, theta_G, mask_F, mask_I):
        global cur_iter, adv_fea
        ms = self.args.mask_size
        c = self.args.channels
        cur_step = self.args.cur_step  #
        args = self.args
        theta_G = torch.nn.Parameter(theta_G, requires_grad=True)
        optimizer = torch.optim.Adam([theta_G], lr=args.cur_step)
        train_epochs = args.maxIter_TG
        epsilon = args.epsilon
        max_p = self.max_p


        for cur_iter in range(train_epochs + 1):
            # print(f"update theta{cur_iter}")
            mask_I.requires_grad = False
            mask_F.requires_grad = False
            theta_G.requires_grad = True
            temp1 = mask_I * mask_F * theta_G
            temp = self.G2I(temp1)
            adv_fea = model(query + temp,
                            input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off  #
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2  #


            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # theta = theta.detach()  #


            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor,
                               args.lr_min)
                self.adjust_learning_rate(optimizer, cur_step)


        # MAPo2t, l11, l22 = return_label(db, dl, adv_fea.detach(), self.target_label, self.kappa)
        # printf(f'{cur_iter}, Theta o2tMAP:{MAPo2t:.5f}')
        return theta_G

    def update_ThetaP(self, query, model, db, dl, tar_fea, tar_label, theta_G, mask_F, mask_I):
        global cur_iter, adv_fea
        ms = self.args.mask_size
        c = self.args.channels
        cur_step = self.args.cur_step  #
        args = self.args
        theta_G = torch.nn.Parameter(theta_G, requires_grad=True)
        optimizer = torch.optim.Adam([theta_G], lr=args.cur_step)
        train_epochs = args.maxIter_TG
        epsilon = args.epsilon
        max_p = self.max_p


        for cur_iter in range(train_epochs + 1):
            # print(f"update theta{cur_iter}")
            mask_I.requires_grad = False
            mask_F.requires_grad = False
            theta_G.requires_grad = True
            temp = mask_I * mask_F * theta_G
            # temp = self.G2I(temp1)
            adv_fea = model(query + temp,
                            input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2


            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # theta = theta.detach()  #


            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor,
                               args.lr_min)
                self.adjust_learning_rate(optimizer, cur_step)


        # MAPo2t, l11, l22 = return_label(db, dl, adv_fea.detach(), self.target_label, self.kappa)
        # printf(f'{cur_iter}, Theta o2tMAP:{MAPo2t:.5f}')
        # theta = torch.clamp(theta_G, -max_p, max_p)
        return theta_G

    # Use the gradient descent method to update F
    def update_Inter(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_I):
        args = self.args
        cur_step = self.args.cur_step
        mask_F = torch.nn.Parameter(mask_F, requires_grad=True)  # Wrap mask F into a tensor so that it can be updated by the optimizer
        optimizer = torch.optim.Adam([mask_F], lr=cur_step)  # Use Adam to optimize mask F

        train_epochs = args.maxIter_FG  #
        for cur_iter in range(1, train_epochs):
            # print(f"update mask_F{cur_iter}")
            mask_I.requires_grad = False
            mask_F.requires_grad = True
            theta.requires_grad = False
            temp1 = mask_I * mask_F * theta
            temp = self.G2I(temp1)
            temp.cuda()

            adv_fea = model(query + temp,
                            input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2

            # Make the group sparse, and then add constraints
            # l3_loss =  (torch.norm(mask_G, p=2) ** 2) * args.lambdaL2
            loss = l1_loss + l2_loss

            optimizer.zero_grad()  # # Clear the gradient cache of the optimizer to zero in to prevent the influence of the previous gradient on this iteration
            loss.backward(retain_graph=True)  # Update the parameters of the model to minimize the parameters of the loss function
            optimizer.step()  # # Update the parameters of the model to minimize the parameters of the loss function
            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)
                self.adjust_learning_rate(optimizer, cur_step)

        mask_F = Variable(mask_F.detach())
        mask_F = self.sort_to_mask(mask_F)
        return mask_F


    # Update the pixel mask I
    def update_Intra(self, query, model, db, dl, tar_fea, tar_label, theta, mask_F, mask_G):
        args = self.args

        cur_step = args.cur_step
        cur_rho1 = args.cur_rho1  # The threshold of perturbation controls the minimum absolute error constraint in the adversarial sample attack, which is controlled within [-cur_rho1, cur_rho1].
        cur_rho2 = args.cur_rho2
        cur_rho3 = args.cur_rho3

        y1 = torch.ones_like(mask_G)
        y2 = torch.ones_like(mask_G)
        z1 = torch.zeros_like(mask_G)
        z2 = torch.zeros_like(mask_G)
        z3 = torch.zeros(1).cuda()
        ones = torch.ones_like(mask_G)
        cur_iter = 0
        err1 = 10
        err2 = 10
        bestMap = 0
        train_epochs = args.maxIter_G

        while (cur_iter < train_epochs + 1):
            # print(f"update mask_I{cur_iter}")
            # mask_I.requires_grad = True
            mask_G.requires_grad = True
            mask_F.requires_grad = False
            theta.requires_grad = False

            # y 1 and y 2 are the results obtained by constraining mask I through the soft threshold function, which are used to meet the constraints of the ADMM algorithm.
            y1 = torch.clamp((mask_G.detach() + z1 / cur_rho1), 0.0, 1.0)  # # Limit the y1 tensor to between 0 and 1.

            # y2 is an intermediate variable used to project into the range [0, 1].
            y2 = self.project_shifted_lp_ball(mask_G.detach() + z2 / cur_rho2, 0.5 * torch.ones_like(mask_G))

            # 3.update mask_b
            # cnn_grad_mask_b
            temp1 = mask_G * mask_F * theta
            temp = self.G2I(temp1)
            temp.cuda()

            # args.lambdaL2 = 0
            # # Use the noisy query image to generate a corresponding eigenvector, and input norm=True means that both the input query and noise are normalized.
            adv_fea = model(query + temp, input_norm=True)
            l1_loss = self.fea_distance(adv_fea, tar_fea) * args.trade_off  # Used to weigh the importance of different goals or constraints.
            l2_loss = (torch.norm(temp, p=2) ** 2) * args.lambdaL2
            l3_loss = (torch.norm(mask_G, p=2)) * args.lambdaL3
            loss = l1_loss + l2_loss + l3_loss

            loss.backward(retain_graph=True)

            cnn_grad_mask_b = mask_G.grad  # Save the gradient of mask I
            mask_b_item = mask_G.sum()  # Find the sum of the element codes in mask I



            grad_mask_loss1 = cnn_grad_mask_b * args.trade_off  # A multiplication of the gradient with the hyperparameters is used to calculate the gradient loss of the mask term.
            grad_mask_loss2 = 2 * mask_G * theta * theta * mask_F * mask_F * args.lambdaL2
            grad_mask_b3 = cur_rho1 * (mask_G - y1)  # The function of this line of code is to calculate the grad_mask_b3, which is a gradient that is used to update the Lagrange multiplier
            grad_mask_b4 = cur_rho2 * (mask_G - y2)  # The function of this line of code is to calculate the grad_mask_b4, which is a gradient that is used to update the Lagrange multiplier

            # The code here is to calculate the gradient corresponding to the Lagrangian multiplier of the mask term, specifically, grad mask b 5 is used to calculate the gradient of the Lagrangian multiplier subterm grad mask loss 5
            grad_mask_b5 = cur_rho3 * (mask_b_item - args.k) * ones

            grad_mask_b = grad_mask_loss1 + grad_mask_loss2 + (z1 + grad_mask_b3) + (z2 + grad_mask_b4) + (
                    z3 * ones + grad_mask_b5)

            mask_G = mask_G - cur_step * grad_mask_b  # What this line of code does is update mask I so that it moves in the direction of minimizing loss.
            # printf([cnn_grad_mask_b[0][i].detach().cpu().numpy().sum() for i in range(16)])
            mask_G = mask_G.detach()
            z1 = z1 + cur_rho1 * (mask_G - y1)  # Update z 1,
            z2 = z2 + cur_rho2 * (mask_G - y2)
            z3 = z3 + cur_rho3 * (mask_G.sum() - args.k)  # Set the sparsity  k = 3*16*112*112
            if cur_iter % args.rho_increase_step == 0:
                # cur_rho1 is updated to increase the result by a certain coefficient and is limited to a maximum value of args.rho1_max.
                cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
                cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
                cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)

            # updating learning rate
            if cur_iter % args.lr_decay_step == 0:
                # ccur_step Either decay by args.lr_decay_factor or remain unchanged (if it is less than args.lr_min after decay).
                cur_step = max(cur_step * args.lr_decay_factor, args.lr_min)

            mask_I = self.G2I(mask_G)
            if cur_iter % args.tick_loss_T == 0 and cur_iter % 1000 == 0:
                with torch.no_grad():  # Disable gradients
                    test_tmp1 = (mask_G.detach() > 0.5).float() * mask_F * theta
                    test_tmp = self.G2I(test_tmp1)
                    test_adv_fea = model(query + test_tmp)

                    # Calculate similarity
                    # MAP_ = compute_MAP(db, dl, test_adv_fea.detach(), tar_label)

            if np.isnan(loss.detach().cpu().numpy()):
                exit(0)
            cur_iter += 1

        # print(torch.mean(mask_G))
        # print(np.sort(mask_G.detach().cpu().numpy()))
        # print(np.mean(mask_G.detach().cpu().numpy()))
        #
        mask_G = (mask_G > 0.5).float().detach()

        return mask_G


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

        mask_F_sort, mask_F_index = torch.norm(mask_F, p=2, dim=[1, 3]).sort(dim=1, descending=True)
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

    def G2I(self, G):

        image_size = self.args.imgsize
        maskSize = self.args.mask_size
        channels = self.args.channels
        query_frame = self.args.num_frames
        stride = self.args.stride

        P = np.floor((image_size - maskSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P
        G = torch.squeeze(G)
        c = torch.unsqueeze(G, 2)
        C = c.expand(-1, -1, maskSize * maskSize, -1)
        M = C.reshape(channels, -1, P * Q)
        mask_Ic = torch.nn.functional.fold(M, output_size=(112, 112), kernel_size=(maskSize, maskSize),
                                           stride=(stride, stride))
        # print("它是可行的！")
        I = torch.unsqueeze(mask_Ic, 0)
        I = I.float()


        return I

    def fileExist(self, name):
        flag = False
        import os
        import os.path

        folder_path = self.args.log_file
        mat_file_name = name
        mat_file_path = os.path.join(folder_path, mat_file_name)

        if os.path.exists(mat_file_path):
            flag = True
        return flag

    def getMask_I(self, mask_IG1):
        import scipy.io as sio
        import glob
        import os.path as osp
        channels = self.args.channels
        query_frame = self.args.num_frames
        path1 = self.args.G_path
        datasetsize = f'{self.args.dataset_name}_1111'
        root_folder = f'/mnt/nvme2n1/zhanyu/zycode/server-HPV-attack/exp/{path1}/{datasetsize}'
        file_folder = f't_net_{self.args.t_net}_{self.args.lossname}/s_net_{self.args.s_net}_512/{self.args.attack_method}'
        # file_folder = f't_net_{self.args.t_net}_{self.args.lossname}/s_net_{self.args.s_net}_512/{self.args.attack_method}/{int(self.args.k/10000)}_{self.args.n}'
        images_list = glob.glob(osp.join(root_folder, file_folder, f'*.mat'))
        for im in images_list:
            if path in im:
                mask_I_path = im
        mat_file = sio.loadmat(mask_I_path)
        mask_I_nd = mat_file['mask_I']
        mask_IG1 = torch.Tensor(mask_I_nd)
        return mask_IG1

    def getMask_F(self, mask_FG1):
        import scipy.io as sio
        import glob
        import os.path as osp
        channels = self.args.channels
        query_frame = self.args.num_frames
        datasetsize = f'{self.args.dataset_name}_1111'
        root_folder = f'/mnt/nvme2n1/zhanyu/zycode/server-HPV-attack/exp/pixel_GPspa5/{datasetsize}'
        file_folder = f't_net_{self.args.t_net}_{self.args.lossname}/s_net_{self.args.s_net}_512/{self.args.attack_method}'
        images_list = glob.glob(osp.join(root_folder, file_folder, f'*.mat'))
        for im in images_list:
            if path in im:
                mask_F_path = im
        mat_file = sio.loadmat(mask_F_path)
        mask_F_nd = mat_file['mask_F']
        mask_FG1 = torch.Tensor(mask_F_nd).cuda()
        return mask_FG1

    def getI_grad(self, mask_IS):
        import scipy.io as sio
        import glob
        import os.path as osp
        channels = self.args.channels
        query_frame = self.args.num_frames
        path1 = self.args.G_path
        datasetsize = f'{self.args.dataset_name}_{self.args.s_datascale}'
        root_folder = f'/mnt/nvme2n1/zhanyu/zycode/server-HPV-attack/exp/{path1}/{datasetsize}'
        file_folder = f't_net_{self.args.t_net}_{self.args.lossname}/s_net_{self.args.s_net}_512/{self.args.attack_method}'
        # file_folder = f't_net_{self.args.t_net}_{self.args.lossname}/s_net_{self.args.s_net}_512/{self.args.attack_method}/{int(self.args.k/10000)}_{self.args.n}'
        images_list = glob.glob(osp.join(root_folder, file_folder, f'*.mat'))
        for im in images_list:
            if path in im:
                mask_F_path = im
        mat_file = sio.loadmat(mask_F_path)
        mask_IS = torch.from_numpy(mat_file['mask_IS'])
        return mask_IS

    @staticmethod
    def project_shifted_lp_ball(x, shift_vec):
        shift_x = x - shift_vec
        # compute L2 norm: sum(abs(v)^2)^(1/2)
        norm2_shift = torch.norm(shift_x, p=2)  # Compute the 2-norm of shift_x.
        n = float(x.numel())  # Returns the total number of elements in the tensor.
        xp = (n ** (1 / 2) / 2) * (shift_x / norm2_shift) + shift_vec
        return xp

    @staticmethod
    def errorCoef(I, Y):
        Z = I.detach().cpu()
        C = Y.detach().cpu()
        err = torch.sum(torch.abs(Z - C) / Z.numel())
        return err
