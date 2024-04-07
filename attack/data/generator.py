import numpy as np
import random
import os


def txt_process(txt_list, parser):
    video_path_list = []
    video_label_list = []
    video_prefix = os.path.join(parser.data_root, 'raw-data')
    with open(txt_list) as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            v_id, label, video_subpath = line.split()
            video_path = os.path.join(video_prefix, video_subpath)
            video_path_list.append(video_path)
            video_label_list.append(int(label))
    return np.array(video_label_list), np.array(video_path_list)


def get_list(parser):
    txt_list_train = None
    txt_list_eval = None
    if parser.dataset == 'UCF101':
        txt_list_train = os.path.join('./dataset/UCF101', 'raw', 'list_cvt', 'trainlist01.txt')
        txt_list_eval = os.path.join('./dataset/UCF101', 'raw', 'list_cvt', 'testlist01.txt')
    elif parser.dataset == 'HMDB51':
        txt_list_train = os.path.join('./dataset/UCF101', 'raw', 'list_cvt', 'trainlist01.txt')
        txt_list_eval = os.path.join('./dataset/UCF101', 'raw', 'list_cvt', 'testlist01.txt')
    kwargs = {}
    kwargs['database_label'], kwargs['database_path'] = txt_process(txt_list_train, parser)
    kwargs['test_label'], kwargs['test_path'] = txt_process(txt_list_eval, parser)
    return kwargs


def generator_list(parser):
    db_lab_path = {}
    kwargs = get_list(parser)
    for l in range(kwargs['database_label'].size):
        db_q = kwargs['database_label'][l]
        db_p = kwargs['database_path'][l]
        if db_lab_path.get(db_q) == None:
            db_lab_path[db_q] = []
            db_lab_path[db_q].append(db_p)
        else:
            db_lab_path[db_q].append(db_p)

    test_lab_path = {}
    for l in range(kwargs['test_label'].size):
        test_q = kwargs['test_label'][l]
        test_p = kwargs['test_path'][l]
        if test_lab_path.get(test_q) == None:
            test_lab_path[test_q] = []
            test_lab_path[test_q].append(test_p)
        else:
            test_lab_path[test_q].append(test_p)

    query_label = []
    query_path = []
    target_path = []
    target_label = []

    random_list = []
    for item in test_lab_path.keys():  # collect dict key'
        random_list.append(item)

    random.shuffle(random_list)

    for qlabels in random_list:
        tarcan_index = [index for index in db_lab_path.keys() if qlabels != index]
        query_candidates = test_lab_path[qlabels]

        t_label = []
        t_path = []
        q_path = []
        q_label = []
        for i in range(parser.n_query):
            tlabels = np.random.choice(np.array(tarcan_index), parser.target_number, replace=False)
            for t_index in range(parser.target_number):
                target_candidates = db_lab_path[tlabels[t_index]]
                t_label.append(tlabels[t_index])
                t_path.append(target_candidates[np.random.choice(len(target_candidates), 1, replace=False)[0]])
            queryChoose_indexes = np.random.choice(len(query_candidates), parser.n_query, replace=False)
            q_label.append(qlabels)
            q_path = [i for i in np.array(query_candidates)[queryChoose_indexes].tolist()]
        target_path.append(t_path)
        target_label.append(t_label)
        query_path.append(q_path)
        query_label.append(q_label)

    target_label = [int(token) for st in target_label for token in st]
    target_path = [token for st in target_path for token in st]
    query_label = [int(token) for st in query_label for token in st]
    query_path = [token for st in query_path for token in st]

    return target_path, target_label, query_path, query_label
