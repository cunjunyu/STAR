import os
import pickle
import random
import time
import numpy as np
import torch
import pandas as pd
import os
import cv2
from copy import deepcopy
from torch import nn
from src.SDD_Dataloader import load_SDD,downsample_all_frame,split_fragmented,traject_preprocess_SDD,SDD_Dataloader
torch.manual_seed(0)
# todo 测试只给了5个，总的数据集有8个，有部分只用作训练？ 命名和实际部分不太符合
DATASET_NAME_TO_NUM = {
    'eth': 0,
    'hotel': 1,
    'zara1': 2,
    'zara2': 3,
    'univ': 4
}

class ETHUCY_Dataloader():
    def __init__(self,args):
        self.args = args
        self.train_data_file = os.path.join(self.args.model_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.model_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.model_dir, "train_batch_cache.cpkl")
        # -----meta-----------
        self.train_seti_batch_cache = os.path.join(self.args.model_dir, "train_seti_batch_cache.cpkl")
        self.train_meta_batch_cache = os.path.join(self.args.model_dir, "train_meta_batch_cache.cpkl")
        # -----MVDG-------------
        self.train_MVDG_batch_cache = os.path.join(self.args.model_dir, "train_MVDG_batch_cache.cpkl")
        if self.args.dataset == 'eth5':
            self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                              './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                              './data/ucy/univ/students001', './data/ucy/univ/students003',
                              './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']

            # Data directory where the pre-processed pickle file resides
            self.data_dir = './data'
            #  每个场景中注释的帧间隔
            skip = [6, 10, 10, 10, 10, 10, 10, 10]

            train_set = [i for i in range(len(self.data_dirs))]
            # 断言 确认相应的test-set在已有数据集中 检查代码中使用的数据集名称是否正确。
            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)
            # 将其转换为数字形式
            args.test_set = DATASET_NAME_TO_NUM[args.test_set]

            if args.test_set == 4 or args.test_set == 5:
                self.test_set = [4, 5]
            else:
                self.test_set = [self.args.test_set]
            # 分离train和test数据集
            for x in self.test_set:
                train_set.remove(x)
            # 获取对应的train和test数据集的地址以及skip 后续应该需要添加对应的val，也可以从train中抽取一部分作为val
            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]
            print("Creating pre-processed data from eth-ucy raw data.")
            # 处理train和test的数据，相应的得到frameped_dict[每帧包含的行人数]和pedtrajec_dict[每个行人包含的单独轨迹数据]
            # 此处返回的是完整的 8个场景的处理完的所有数据
            if not (os.path.exists(self.train_data_file) and os.path.exists(self.test_data_file)):
                self.traject_preprocess('train')
                self.traject_preprocess('test')
            print("Done.")

        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict,self.test_scene_list = self.load_dict(self.test_data_file)
            self.dataPreprocess('test',dataset=self.test_dataset)
        self.testbatch, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache)





