import os
import time
import socket
import logging
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from utils.evaluation import *
"""
Static Model
"""
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 dataset=None,
                 hash_bit = None,
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))
        self.dataset = dataset
        self.hash_bit = hash_bit
        # init params
        self.net = net
        self.model_prefix = model_prefix
        self.margin = 12

    def load_state(self, state_dict, strict=False):
        if strict:
            self.net.load_state_dict(state_dict=state_dict)
        else:
            # customized partialy load function
            net_state_keys = list(self.net.state_dict().keys())
            for name, param in state_dict.items():
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)
            # indicating missed keys
            if net_state_keys:
                logging.warning(">> Failed to load: {}".format(net_state_keys))
                return False
        return True

    def get_checkpoint_path(self, epoch):
        assert self.model_prefix, "model_prefix undefined!"
        checkpoint_path = "./exp/{}/{}ep-{:04d}_{}.pth".format(self.model_prefix,self.model_prefix, epoch, self.dataset)

        return checkpoint_path


    def save_checkpoint(self, epoch):

        save_path = self.get_checkpoint_path(epoch)
        save_folder = os.path.dirname(save_path)

        if not os.path.exists(save_folder):
            logging.debug("mkdir {}".format(save_folder))
            os.makedirs(save_folder)
        torch.save(self.net.state_dict(), save_path)
        logging.info("Checkpoint (only model) saved to: {}".format(save_path))

    def similarity(self,label1,label2):
        return label1==label2 # default with singe label

    def tripletLoss(self,x,labels):
        triplet_loss = torch.tensor(0.0).cuda()
        labels_ = labels.cpu().data.numpy()
        triplets = []
        for label in labels_:
            label_mask = (labels_ == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            if len(negative_indices) < 1:
                continue
            anchor_positives = list(itertools.combinations(label_indices, 2))
            temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                    for neg_ind in negative_indices]
            triplets += temp
        if triplets:
            triplets = np.array(triplets)
            # print triplets
            sq_ap = (x[triplets[:, 0]] - x[triplets[:, 1]]).pow(2).sum(1)
            sq_an = (x[triplets[:, 0]] - x[triplets[:, 2]]).pow(2).sum(1)
            losses = F.relu(self.margin + sq_ap - sq_an)
            triplet_count = torch.tensor(losses.size()[0]).float().cuda()
            if triplet_count > 0:
                triplet_loss = losses.sum() / triplet_count

        return triplet_loss

    def forward(self, data, target):
        data = data.float().cuda()
        target = target.cuda()

        embeddings = self.net(data)
        loss       = self.tripletLoss(embeddings,target)
        return [embeddings], [loss]



"""
Dynamic model that is able to update itself
"""
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback_freq=10,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 dataset = None,
                 hash_bit = None,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion, model_prefix=model_prefix)
        self.dataset = dataset
        self.hash_bit = hash_bit


        self.step_callback_freq = step_callback_freq
        self.save_checkpoint_freq = save_checkpoint_freq
        self.batch_size=opt_batch_size
        self.margin = 12


    """
    Optimization
    """
    def fit(self,
            train_iter,
            optimizer,
            lr_scheduler,
            eval_iter=None,
            epoch_start=0,
            epoch_end=160,
            **kwargs):

        """
        checking
        """
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"

        maxMAP = 0

        for i_epoch in range(epoch_start, epoch_end):
            epoch_start_time = time.time()

            self.net.train()
            sum_sample_inst = 0
            sum_sample_elapse = 0.
            sum_update_elapse = 0
            batch_start_time = time.time()
            logging.info("Start epoch {:d}:".format(i_epoch))



            for i_batch, (data, target, _) in enumerate(train_iter):

                update_start_time = time.time()

                # [forward] making next step
                outputs, losses = self.forward(data,target)  # output: [batch_size, 64]

                # [backward]
                optimizer.zero_grad()
                for loss in losses:
                    loss.backward()
                optimizer.step()
                # timing each batch
                sum_sample_elapse += time.time() - batch_start_time
                sum_update_elapse += time.time() - update_start_time
                batch_start_time = time.time()
                sum_sample_inst += data.shape[0]


                if (i_batch % self.step_callback_freq) == 0:
                    logging.info("epoch:[{}] batch:[{}] cost time:[{:.02f}] loss:[{}]".format(
                        i_epoch,i_batch,sum_sample_elapse/self.step_callback_freq,losses[0].detach().cpu().mean()))
            lr_scheduler.step()


            if i_epoch % 5 == 0:  # (epoch + 1) % 2 == 0:
                logging.info("Start evaluating epoch {:d}:".format(i_epoch))
                self.net.eval()
                db_binary, db_label     = inference(train_iter,  self.net)
                test_binary, test_label = inference(eval_iter,   self.net)
                MAP_ = compute_MAP(db_binary, db_label, test_binary, test_label)

                logging.info("epoch:{}  MAP_:{} ".format(i_epoch+1,MAP_))
                if MAP_ > maxMAP:
                    maxMAP = MAP_
                    self.save_checkpoint(epoch=i_epoch + 1)

        logging.info("Optimization done!")

