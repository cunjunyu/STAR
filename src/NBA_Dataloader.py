from torch.utils.data import Dataset
import torch
import pickle
import random
import time
import pandas as pd
import os
import cv2
from copy import deepcopy
from torch import nn
import numpy as np

def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'nba',
    }

    return data

DATASET_NAME_TO_NUM = {
    'CLE': 0,
    'GSW': 1,
    'NYK': 2,
    'OKC': 3,
    'SAS': 4,
}

class NBA_Dataloader():
    def __init__(self, args):
        # dataset : NBA
        # data_dirs : CLE , GSW , OKC , SAS
        self.args = args
        if self.args.dataset == 'NBA':
            self.args.seq_length = 15
            self.args.obs_length = 5
            self.args.pred_length = 10
            if args.test_set in ['CLE','GSW','NYK','OKC','SAS']:
                self.data_dirs = ['./data/NBA/nba/source/CLE', './data/NBA/nba/source/GSW', './data/NBA/nba/source/NYK','./data/NBA/nba/source/OKC', './data/NBA/nba/source/SAS']

                # Data directory where the pre-processed pickle file resides
                # self.data_dir = './data'
                skip = [10, 10, 10, 10, 10 ]

                train_set = [i for i in range(len(self.data_dirs))]

                assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)

                test_set = DATASET_NAME_TO_NUM[args.test_set]

                # if args.test_set == 4 or args.test_set == 5:
                #     self.test_set = [4, 5]
                # else:
                self.test_set = [test_set]

                for x in self.test_set:
                    train_set.remove(x)

                self.train_dir = [self.data_dirs[x] for x in train_set]
                self.test_dir = [self.data_dirs[x] for x in self.test_set]
                self.trainskip = [skip[x] for x in train_set]
                self.testskip = [skip[x] for x in self.test_set]
            elif args.test_set == 'nba':
                #  相对应的为nba的原始划分数据 需要将其处理成一样的格式 即后期需要将原本的训练集按4份随机划分即可
                self.train_dir = './data/NBA/nba/train.npy'
                self.test_dir = './data/NBA/nba/test.npy'
                self.trainskip = [10,10,10,10]
                self.testskip = [10]
        else:
            raise NotImplementedError

        self.train_data_file = os.path.join(self.args.model_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.model_dir, "test_trajectories.cpkl")

        self.test_batch_cache = os.path.join(self.args.model_dir, "test_batch_cache.cpkl")
        self.test_batch_cache_split = os.path.join(self.args.model_dir, "test_batch_cache_0.cpkl")

        self.train_batch_cache = os.path.join(self.args.model_dir, "train_batch_cache.cpkl")
        self.train_batch_cache_split = os.path.join(self.args.model_dir, "train_batch_cache_0.cpkl")

        self.train_MLDG_batch_cache = os.path.join(self.args.model_dir, "train_MLDG_batch_cache.cpkl")
        self.train_MLDG_batch_cache_split = os.path.join(self.args.model_dir, "train_MLDG_batch_cache_0.cpkl")

        self.train_MVDG_batch_cache = os.path.join(self.args.model_dir, "train_MVDG_batch_cache.cpkl")
        self.train_MVDG_batch_cache_split = os.path.join(self.args.model_dir, "train_MVDG_batch_cache_0.cpkl")

        print("Creating pre-processed data from NBA  raw data.")
        self.traject_preprocess('train')
        self.traject_preprocess('test')
        print("Done.")

        # Load the processed data from the pickle file
        print("process test data ")
        if not (os.path.exists(self.test_batch_cache) or os.path.exists(self.test_batch_cache_split)):
            self.test_traject_dict = self.load_dict(self.test_data_file)
            self.dataPreprocess('test')
        self.testbatch, self.testbatchnums = self.load_split_data(self.test_batch_cache)
        print('Total number of test batches:', self.testbatchnums)

        if self.args.stage == 'origin':
            print("Preparing origin data batches.")
            if not (os.path.exists(self.train_batch_cache) or os.path.exists(self.train_batch_cache_split)):
                self.traject_dict = self.load_dict(self.train_data_file)
                self.dataPreprocess('train')
            self.trainbatch, self.trainbatchnums = self.load_split_data(self.train_batch_cache)
            print('Total number of training origin batches:', self.trainbatchnums)

        elif self.args.stage == 'meta':
            if not (os.path.exists(self.train_MLDG_batch_cache) or os.path.exists(self.train_MLDG_batch_cache_split)):
                print("process train meta cpkl")
                self.traject_dict= self.load_dict(self.train_data_file)
                self.MLDG_TASK(setname='train')
            self.train_batch_task, self.train_batch_tasknums = self.load_split_data(self.train_MLDG_batch_cache)
            print('Total number of training MLDG task batches :', len(self.train_batch_task))

        elif self.args.stage == 'MVDG' or self.args.stage =='MVDGMLDG':
            if not (os.path.exists(self.train_MVDG_batch_cache) or os.path.exists(self.train_MVDG_batch_cache_split)):
                self.traject_dict= self.load_dict(self.train_data_file)
                self.MVDG_TASK(setname='train')
            self.train_batch_MVDG_task,self.train_batch_MVDG_tasknums = self.load_split_data(self.train_MVDG_batch_cache)
            print('Total number of training MVDG task batches :', len(self.train_batch_MVDG_task))

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)    
    
    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        traject_dict = raw_data

        return traject_dict


    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data
    # 针对于pickle data的数据量较大的问题 进行分析 运用拆包技术 打开
    def pick_split_data(self,trainbatch,cachefile,batch_size):
        # todo 注意此处的cachefile 与原本的不一样 可能会引起命名的不一 可以从已有的cachafile中提取对应的字符与其结合
        # 第一步：提取相应的名称作为分批前缀
        # 删除文件扩展名 前面有一个./ 后续也有相应的。cpkl
        filename_without_extension, file_extension = os.path.splitext(cachefile)
        # 第二步：分割大型对象
        num_batches = len(trainbatch) // batch_size  # 计算总批次数
        remainder = len(trainbatch) % batch_size  # 剩余的元素数量
        for i in range(num_batches):
            # 计算分批的起始索引和结束索引
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            # 获取当前批次的子集
            batch_train = trainbatch[start_idx:end_idx]
            # 执行序列化操作 格式化字符串（f-string）来创建文件名，其中 {i} 表示当前循环的索引值
            cachefile = f"{filename_without_extension}_{i}.cpkl"
            print('pickle data to '+str(cachefile))
            with open(cachefile, "wb") as f:
                pickle.dump(batch_train, f, protocol=2)
                f.close()
        # 处理剩余的元素（如果有）
        if remainder > 0:
            start_idx = num_batches * batch_size
            end_idx = start_idx + remainder
            batch_train = trainbatch[start_idx:end_idx]
            cachefile = f"{filename_without_extension}_{num_batches}.cpkl"
            print('pickle data to ' + str(cachefile))
            with open(cachefile, "wb") as f:
                pickle.dump(batch_train, f, protocol=2)
                f.close()
            num_batches += 1


    def load_split_data(self,cachefile):
        # batch_size = 1000  # 每个批次的大小
        # num_batches = 10  # 分批的总数
        # 第一步 判断对应的存储文件是简单的cachefile还是复杂的cachefile——0
        filename_without_extension, file_extension = os.path.splitext(cachefile)
        cachefile_split = f"{filename_without_extension}_{0}.cpkl"
        if os.path.exists(cachefile):
            f = open(cachefile, 'rb')
            raw_data = pickle.load(f)
            f.close()
            return raw_data
        elif os.path.exists(cachefile_split):
            # 第二步 统计每个目录下有多少个分开的文件夹
            # 去除最后一个部分，只保留前面的部分 即上级目录
            directory_path = os.path.dirname(cachefile_split)
            # 选取相应的文件夹中的前缀名 eg:test_batch_cache
            file_prefix = filename_without_extension.split('/')[-1]
            # directory_path,file_prefix = os.path.split(cachefile_split)
            file_list = [filename for filename in os.listdir(directory_path) if filename.startswith(file_prefix) and filename.endswith(file_extension)]
            num_batches = len(file_list)
            combined_trainbatch = []
            #第三步 分文件按顺序读出文件 并拼接
            for i in range(num_batches):
                cachefile = f"{filename_without_extension}_{i}.cpkl"
                print('load data from :'+str(cachefile))
                with open(cachefile, "rb") as f:
                    batch_train = pickle.load(f)
                    f.close()
                combined_trainbatch.extend(batch_train)
            # 将分批的子集重新组合成完整的对象
            trainbatch = combined_trainbatch
            trainbatch_nums = len(trainbatch)
            return trainbatch, trainbatch_nums
        else:
            raise NotImplementedError


    def get_train_batch(self, idx):
        batch_data, batch_id = self.trainbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)

        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.testbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False)
        return batch_data, batch_id
    
    def reset_batch_pointer(self, set, valid=False):
        '''
        Reset all pointers
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0
            
    def traject_preprocess(self, setname):
        '''
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:
            data_dirs = self.test_dir
            data_file = self.test_data_file
        trajec_dict = {}
        # For each dataset
        if self.args.test_set in ['CLE','GSW','NYK','OKC','SAS']:
            for seti, directory in enumerate(data_dirs):
                all_trajs_data_root = os.path.join(directory, "all_trajs.npy")
                all_teams_data_root = os.path.join(directory, "all_teams.npy")

                self.all_trajs = np.load(all_trajs_data_root)
                self.all_trajs /= (94/28) # Turn to meters

                self.all_teams = np.load(all_teams_data_root)

                self.all_trajs = self.all_trajs.transpose(0,2,1,3)
                self.all_teams = self.all_teams.transpose(0,2,1)
                # all_trajs [803,10,15,2] [B,ped-num,time-seq,xy]
                # all_teams [1,10,15] [1?,ped-num,time-seq]

                trajec_dict[seti] = self.all_trajs
            ## 写一个[seti][ped][traj]
            # 在test为CLE的情况下，'GSW': 803  'OKC': 705  ‘SAS’ ：895, 'CLE' : 717
        elif self.args.test_set == 'nba':
            self.all_trajs = np.load(data_dirs)
            self.all_trajs /= (94 / 28)  # Turn to meters
            self.all_trajs = self.all_trajs.transpose(0, 2, 1, 3)
            # 注意需要删除对应的球的数据 因为这里写死了2类 则球无 先去除其数据 然后发现 球是最后一个
            self.all_trajs = self.all_trajs[:,:-1]
            if setname == 'train':
                # training完整的37500，又只取前32500，剩余5000？
                self.all_trajs = self.all_trajs[:32500]
                # 依据对应的划分 将其划分到相应的4个部分 使其符合后期的数据分析
                # 定义均分的数量
                num_parts = 4
                # 计算每个部分的大小
                part_size = self.all_trajs.shape[0] // num_parts

                # 均分并存储到 dict 中
                for i in range(num_parts):
                    start_idx = i * part_size
                    end_idx = (i + 1) * part_size
                    part_data = self.all_trajs[start_idx:end_idx]
                    trajec_dict[i] = part_data
            elif setname == 'test':
                # testing 完整的为47434 只取12500？
                self.all_trajs = self.all_trajs[:12500]
                trajec_dict[0] = self.all_trajs
        f = open(data_file, "wb")
        pickle.dump(trajec_dict, f, protocol=2)
        f.close()
        
    def get_data_index(self, data_dict, setname, ifshuffle=True):
        '''
        Get the dataset sampling index.
        '''
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        for item in data_dict.items():
            # todo 此处与原有的代码不一样 注意！! 应该是针对于NBA的处理特殊？
            key = item[0]
            value = item[1]
            frames = value
            # --------一样
            # 此处其实不是frmae 准确的说应该是片段
            total_frame += len(frames)
            set_id.extend(list(key for i in range(len(frames))))
            frame_id_in_set.extend(list(i for i in range(len(frames))))

        all_frame_id_list = list(i for i in range(total_frame))

        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # to make full use of the data
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def get_data_index_meta(self, seti, data_dict, setname, ifshuffle=True):
        """
        输入的data-dict是个list，只有单个场景的数据，时间序列，其存储了每个场景从第一帧到最后一帧（固定间隔）的行人标号
        setname：train/test

        """
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        # 此处其实不是frmae 准确的说应该是片段
        total_frame += len(data_dict)
        set_id.extend(list(seti for i in range(len(data_dict))))
        frame_id_in_set.extend(list(i for i in range(len(data_dict))))
        all_frame_id_list = list(i for i in range(total_frame))
        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # to make full use of the data
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname == 'train':
            # 不一样 没有对应的frame-list与ped-list
            traject_dict = self.traject_dict
            cachefile = self.train_batch_cache
        else:
            traject_dict = self.test_traject_dict
            cachefile = self.test_batch_cache
            
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        data_index = self.get_data_index(traject_dict, setname, ifshuffle=shuffle)
        # traject——dict不一样
        trainbatch = self.get_seq_from_index_balance(traject_dict, data_index, setname)
        trainbatchnums = len(trainbatch)
        print('NBA batch_nums:'+str(trainbatchnums))
        if trainbatchnums < 50:
            f = open(cachefile, "wb")
            pickle.dump((trainbatch, trainbatchnums), f, protocol=2)
            f.close()
        else:
            self.pick_split_data(trainbatch, cachefile, batch_size=50)

    def MLDG_TASK(self,setname):
        # 还未验证 需要后续分析
        # 第一步 加载对应数据集以及相应参数
        traject_dict = self.traject_dict
        cachefile = self.train_MLDG_batch_cache
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        trainbatch_meta = []
        trainbatchnums_meta = []
        # 注意此处的traject-dict为一个dict，dict中包含相应的4个ndarray，对其迭代不能使用enumerate ，该函数适用于迭代列表，元组或其他可迭代对象，字典用items
        # 第二步 按场景分解获取对应batch数据
        for seti,seti_traject_dict in traject_dict.items():
            trainbatch_meta.append({})
            # data-index 处按相应的场景分开即可
            data_index = self.get_data_index_meta(seti,seti_traject_dict,setname,ifshuffle=shuffle)
            train_index = data_index
            train_batch = self.get_seq_from_index_balance(traject_dict,train_index, setname)
            trainbatchnums = len(train_batch)
            # list（场景号） - list（windows号） -tuple （）
            trainbatch_meta[seti] = train_batch
            trainbatchnums_meta.append(trainbatchnums)
        task_list = []
        for seti, seti_batch_num in enumerate(trainbatchnums_meta):
            if trainbatchnums_meta[seti] == 0 or []:
                continue
            query_seti_id = list(range(len(trainbatchnums_meta)))
            # 第一步依据seti以及对应的scene-list找出与set相同的场景，其他不同的加入到query——seti-id里
            query_seti_id.remove(seti)
            for batch_id in range(seti_batch_num):
                support_set = trainbatch_meta[seti][batch_id]
                # support-set 为tuple 包含tuple和list，tuple中又有0,1,2,3个ndarray和1个list
                if len(support_set[0][0]) == 0 or len(support_set) == 0:  # 需要深入分析
                    continue
                for query_i in range(self.args.query_sample_num):
                    random_query_seti = random.choice(query_seti_id)
                    while len(trainbatch_meta[random_query_seti][0][0]) == 0 or len(
                            trainbatch_meta[random_query_seti]) == 0:
                        random_query_seti = random.choice(query_seti_id)
                    random_query_seti_batch = random.randint(0, trainbatchnums_meta[random_query_seti] - 1)
                    query_set = trainbatch_meta[random_query_seti][random_query_seti_batch]
                    task_list.append((support_set, query_set,))
        batch_task =[task_list[i:i + self.args.query_sample_num] for i in
                               range(0, len(task_list), self.args.query_sample_num)]
        batch_task_num = len(batch_task)
        # 此处的batch-task-num  ->  需要调节
        if batch_task_num < 50:
            f = open(cachefile, "wb")
            pickle.dump(batch_task, f, protocol=2)
            f.close()
        else:
            self.pick_split_data(batch_task, cachefile, batch_size=50)


    def MVDG_TASK(self,setname):
        # 还未验证 需要后续分析
        # 第一步 加载对应数据集以及相应参数
        traject_dict = self.traject_dict
        cachefile = self.train_MVDG_batch_cache
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        trainbatch_meta = []
        trainbatchnums_meta = []
        # 注意此处的traject-dict为一个dict，dict中包含相应的4个ndarray，对其迭代不能使用enumerate ，该函数适用于迭代列表，元组或其他可迭代对象，字典用items
        # 第二步 按场景分解获取对应batch数据
        for seti,seti_traject_dict in traject_dict.items():
            trainbatch_meta.append({})
            # data-index 处按相应的场景分开即可
            data_index = self.get_data_index_meta(seti,seti_traject_dict,setname,ifshuffle=shuffle)
            train_index = data_index
            train_batch = self.get_seq_from_index_balance(traject_dict,train_index, setname)
            trainbatchnums = len(train_batch)
            # list（场景号） - list（windows号） -tuple （）
            trainbatch_meta[seti] = train_batch
            trainbatchnums_meta.append(trainbatchnums)
        # 第三步 形成task-list
        task_list = []
        for seti, seti_batch_num in enumerate(trainbatchnums_meta):
            if trainbatchnums_meta[seti] == 0 or []:
                continue
            query_seti_id = list(range(len(trainbatchnums_meta)))
            # 第一步依据seti以及对应的scene-list找出与set相同的场景，其他不同的加入到query——seti-id里
            query_seti_id.remove(seti)
            for batch_id in range(seti_batch_num):
                support_set = trainbatch_meta[seti][batch_id]
                # support-set 为tuple 包含tuple和list，tuple中又有0,1,2,3个ndarray和1个list
                if len(support_set[0][0]) == 0 or len(support_set) == 0:  # 需要深入分析
                    continue
                for query_i in range(self.args.query_sample_num):
                    random_query_seti = random.choice(query_seti_id)
                    while len(trainbatch_meta[random_query_seti][0][0]) == 0 or len(
                            trainbatch_meta[random_query_seti]) == 0:
                        random_query_seti = random.choice(query_seti_id)
                    random_query_seti_batch = random.randint(0, trainbatchnums_meta[random_query_seti] - 1)
                    query_set = trainbatch_meta[random_query_seti][random_query_seti_batch]
                    task_list.append((support_set, query_set,))
        batch_task_list =[task_list[i:i + self.args.query_sample_num] for i in
                               range(0, len(task_list), self.args.query_sample_num)]
        # 第四步：形成mvdg-list
        random.shuffle(batch_task_list)
        # 因为相应的MVDG框架有多条优化轨迹，每条轨迹有多个task；故而batch——task对应没条轨迹，则需要再依轨迹进行聚合
        # 先将batch-task补充到3的倍数
        optim_trajectory_num = self.args.optim_trajectory_num
        num_groups = len(batch_task_list) // optim_trajectory_num
        if len(batch_task_list) % optim_trajectory_num != 0:
            num_groups += 1
        new_batch_task_list = [[] for _ in range(num_groups)]
        for i, item in enumerate(batch_task_list):
            group_index = i // optim_trajectory_num
            new_batch_task_list[group_index].append(item)
        # 补充不完整的数据
        if len(batch_task_list) % optim_trajectory_num != 0:
            remaining = optim_trajectory_num - (len(batch_task_list) % optim_trajectory_num)
            for _ in range(remaining):
                random_index = random.randint(0, len(batch_task_list) - 1)
                new_batch_task_list[-1].append(batch_task_list[random_index])
        print("Finsh MVDG task batch" + str(setname))
        new_batch_task_list_num = len(new_batch_task_list)
        print('mvdg task number : ' + str(new_batch_task_list_num))
        if new_batch_task_list_num < 50 :
            f = open(cachefile, "wb")
            pickle.dump(new_batch_task_list, f, protocol=2)
            f.close()
        else:
            self.pick_split_data(new_batch_task_list,cachefile,batch_size= 50)

    def get_seq_from_index_balance(self, traject_dict, data_index, setname):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''
        batch_data_mass = []
        batch_data = []
        Batch_id = []
        # todo 每段视频下是固定的行人数目 故而整体会简单许多
        present_pedi = 11
        if setname == 'train':
            skip = self.trainskip
        else:
            skip = self.testskip

        ped_cnt = 0
        last_frame = 0
    
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, all_frame = data_index[:, i]
                        
            cur_trajec = traject_dict[cur_set][cur_frame]
            
            cur_trajec = cur_trajec.transpose(1,0,2)
            # 前述步骤不一样 主要是因为相对的人数是固定的 故而简化 目标都是获取相对的 frmea-list，ped-list，2 格式数据
            batch_pednum = sum([i.shape[1] for i in batch_data]) + cur_trajec.shape[1]

            cur_pednum = cur_trajec.shape[1]
            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame,)

            batch_data.append(cur_trajec)
            Batch_id.append(batch_id)
            
            if batch_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                batch_data = self.massup_batch(batch_data)
                if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                    batch_data_mass.append((batch_data, Batch_id,))
                elif batch_data[4] == []:
                    print('舍弃该数值')
                # batch_data_mass.append((batch_data, Batch_id,))

                batch_data = []
                Batch_id = []
                
                last_frame = i

        # if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
        if last_frame < data_index.shape[1] - 1 and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数值')
            # batch_data_mass.append((batch_data, Batch_id,))
        return batch_data_mass

    def massup_batch(self, batch_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        if self.args.dataset == "eth5":
            relation_num = 1
        elif self.args.dataset == "SDD" or self.args.dataset == "NBA" or self.args.dataset == "NFL":
            relation_num = 3
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]

        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 2))
        # 与SDD统一格式 都为 relation seq-length num-peds num-peds
        nei_list_b = np.zeros((relation_num,self.args.seq_length,  num_Peds, num_Peds))
        nei_num_b = np.zeros((relation_num,self.args.seq_length,  num_Peds))

        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            num_Ped = batch.shape[1]
            # todo 重写get-social-inputs-numpy
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch,relation_num)
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, :, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, :, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

    def get_social_inputs_numpy(self, inputnodes,relation_num):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]
        relation_matrix = np.zeros((num_Peds,num_Peds))
        # 单独考虑两队球员的轨迹 todo 在SDD中针对性的改为依据label
        for i in range(num_Peds):
            for j in range(num_Peds):
                # 球队1
                if i < 5 and j < 5:
                    relation_matrix[i][j] = 0
                # 球队2
                elif i >= 5 and j >= 5:
                    relation_matrix[i][j] = 1
                # 球队1和球队2之间的关系
                elif i < 5 and j >= 5:
                    relation_matrix[i][j] = 2
                elif i >= 5 and j < 5:
                    relation_matrix[i][j] = 2
        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((relation_num,inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((relation_num,inputnodes.shape[0], num_Peds))

        for relation in range(relation_num):
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
            cur_nei_list = np.zeros((inputnodes.shape[0],num_Peds, num_Peds))
            cur_nei_num = np.zeros((inputnodes.shape[0],num_Peds))
            for pedi in range(num_Peds):
                cur_nei_list[:, pedi, :] = seq_list
                cur_nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
                cur_nei_num[:, pedi] = np.sum(cur_nei_list[:, pedi, :], 1)    
                seqi = inputnodes[:, pedi]
                for pedj in range(num_Peds):
                    # if relation_matrix[pedi][pedj] != relation:
                    #     cur_nei_list[:, pedi, pedj] = 0
                    #     continue
                    
                    seqj = inputnodes[:, pedj]
                    select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                    relative_cord = seqi[select, :2] - seqj[select, :2]

                    # invalid data index  添加一条，只保存对应类型的行人关系图邻居 根据关系类型来选择性地保存特定类型的行人邻居关系
                    select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                            abs(relative_cord[:, 1]) > self.args.neighbor_thred) | (relation_matrix[pedi][pedj] != relation and pedi != pedj)
                    # 自己与自己需要删除

                    cur_nei_num[select, pedi] -= select_dist

                    select[select == True] = select_dist
                    cur_nei_list[select, pedi, pedj] = 0
            nei_list[relation,:,:,:] = cur_nei_list
            nei_num[relation,:,:] = cur_nei_num
            
        return seq_list, nei_list, nei_num
    
    def rotate_shift_batch(self, batch_data, ifrotate=True):
        '''
        Random ration and zero shifting.
        '''
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data

        # rotate batch
        if ifrotate:
            th = random.random() * np.pi
            cur_ori = batch.copy()
            batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)
        # get shift value
        s = batch[self.args.obs_length - 1]

        shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)

        batch_data = batch, batch - shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum
        return batch_data


class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):
        super(NBADataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if training:
            data_root = 'datasets/nba/train.npy'
        else:
            data_root = 'datasets/nba/test.npy'

        self.trajs = np.load(data_root) 
        self.trajs /= (94/28) # Turn to meters

        if training:
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:12500]

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        out = [past_traj, future_traj]
        return out




