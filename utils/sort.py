from __future__ import print_function, division
import os
import random
import math
import pickle
import json


def find_dict(path="./checkpoint/checkpoints_768",
              dataset='UCF101',
              model_name='C3DFea',
              loss_name='TRIPLETMARGIN16frames',
              embedding_size='768',model_scale=None):
    path = "./checkpoint/checkpoints_"+str(embedding_size)
    dic={}

    txt_list = os.listdir(path)
    for doc in txt_list:
        if '.py'  in doc:
            continue
        dic[doc]={}
        txt_list_model=os.listdir((path+'/'+doc))
        for doc2 in txt_list_model:
            dic[doc][doc2]={}
            txt_list_loss=txt_list_model=os.listdir((path+'/'+doc+'/'+doc2))
            for doc3 in txt_list_loss:
                dic[doc][doc2][doc3] = {}
                txt_list_weight = txt_list_model = os.listdir((path + '/' + doc + '/' + doc2+'/' + doc3))
                max_map=0.0
                dic[doc][doc2][doc3]['model_dict']=""
                # dic[doc][doc2][doc3]['db_binary']=""
                # dic[doc][doc2][doc3]['db_label'] = ""
                # dic[doc][doc2][doc3]['test_binary'] = ""
                # dic[doc][doc2][doc3]['test_label'] = ""
                for weight_name in txt_list_weight:
                    if 'dat' not in weight_name:
                        continue
                    weight=weight_name.split('mAP')[1].split('.pth')[0]
                    if  float(weight)>float(max_map):
                        max_map=weight
                        dic[doc][doc2][doc3]['model_dict'] = weight_name

                for other_name in txt_list_weight:
                    if  str(max_map) not in other_name:
                        continue
                    if 'dat' in other_name:
                        continue
                    if 'db_binary' in other_name :
                        dic[doc][doc2][doc3]['db_binary'] = other_name
                    elif'db_label' in other_name:
                        dic[doc][doc2][doc3]['db_label'] = other_name
                    elif 'test_binary' in other_name:
                        dic[doc][doc2][doc3]['test_binary'] = other_name
                    elif 'test_label' in other_name:
                        dic[doc][doc2][doc3]['test_label'] = other_name
        with open('./model_chaeck.txt', 'w', encoding='utf-8') as f:
            f.write(json.dumps(dic))


        # print(txt_list_loss)
        # print(dic)

    txt_list.sort(key=lambda x: str(x[:-4]))
    # print(txt_list)
    # model_dict = dic[f'{dataset}_{model_scale}'][model_name][loss_name]['model_dict']
    # db_binary  = dic[f'{dataset}_{model_scale}'][model_name][loss_name]['db_binary']
    # db_label   = dic[f'{dataset}_{model_scale}'][model_name][loss_name]['db_label']
    # test_binary= dic[f'{dataset}_{model_scale}'][model_name][loss_name]['test_binary']
    # test_label = dic[f'{dataset}_{model_scale}'][model_name][loss_name]['test_label']

    t_map = dic[f'{dataset}'][model_name][loss_name]['model_dict'].split('mAP')[1].split('.pth')[0]
    return t_map




if __name__=='__main__':
    find_dict(path="../checkpoint/checkpoints_768",
              dataset='UCF101',
              model_name='C3DFea',
              loss_name='TRIPLETMARGINLOSS'.upper()+'16frames',
              model_scale='1111')
