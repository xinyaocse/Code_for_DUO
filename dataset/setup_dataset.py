import  numpy as np
import  os
from tmp.evaluation import *
import argparse
import model.dataset as dataset
from model.data import attack_iterator
from model.data import iterator_factory
from model.network.model import *
from tmp.evaluation import *

def pr_curve(params,test_hash, test_label):
    qF = test_hash
    rF = params['database_hash']
    qL = test_label
    rL = params['database_label']
    topK = params['R']

    # print(np.shape(qF), np.shape(rF))
    qF, rF, qL, rL = np.array(qF), np.array(rF), np.array(qL), np.array(rL)
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]
    Rank = np.argsort(cdist(qF, rF, 'hamming'))
    P, R = [], []
    for k in range(1, topK + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = np.zeros(n_query)  # 各 query sample 的 Precision@R
        r = np.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):
            Gnd = ((qL[it] == rL)>0).astype(np.float32)# 枚举 query sample
            gnd = Gnd
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))
    # print(P)
    # print(R)
    arr = []
    for p, r in zip(P, R):
        arr.append([p, r])
    return np.array(arr)

def precision_curve(params,test_hash, test_label):
    params["step"] = 10
    database_hash = np.array(params['database_hash'])
    validation_code = test_hash
    database_label = np.array(params['database_label'])
    validation_labels = test_label
    query_num = validation_code.shape[0]
    database_hash = np.sign(database_hash)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_hash, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    arr = []

    for iter in range(100):
        R = (iter + 1) * 1
        APx = []
        for i in range(query_num):
            label = validation_labels[i]
            # label[label == 0] = -1
            idx = ids[:, i]
            imatch = np.sum(database_label[idx[0:R]] == label) > 0
            relevant_num = np.sum(imatch)
            APx.append(float(relevant_num) / R)
        arr.append([R, np.mean(np.array(APx))])
    return np.array(arr)


def setup_dataset(args):
    file_dir = os.path.join(args['basepath'], 'model', 'dataset', args['dataset'],args['network'] + '_' + str(args['hash_bit']) + 'bits')
    print('loading binary code and label')
    args['database_hash'] = np.load(os.path.join(file_dir, 'database_hash.npy'), allow_pickle=True)
    args['database_label'] = np.load(os.path.join(file_dir, 'database_label.npy'), allow_pickle=True)
    args['database_path'] = np.load(os.path.join(file_dir, 'database_path.npy'), allow_pickle=True)
    args['test_hash'] = np.load(os.path.join(file_dir, 'test_hash.npy'), allow_pickle=True)
    args['test_label'] = np.load(os.path.join(file_dir, 'test_label.npy'), allow_pickle=True)
    args['test_path'] = np.load(os.path.join(file_dir, 'test_path.npy'), allow_pickle=True)
    # args['R'] = 608

    # generate database


    dataset_cfg,args = dataset.get_config(name=args['dataset'],args=args)
    args.update(dataset_cfg)
    MAP, APx = mean_average_precision(args, args['test_hash'], args['test_label'])
    print(args['pretrained_3d'], 'MAP: %.6f' % MAP)

    return args


