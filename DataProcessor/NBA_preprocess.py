import os
import numpy as np
import pickle
import random
from torch.utils.data import Dataset
import torch

from DataProcessor.DataProcessorFactory import DatasetProcessor_BASE


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

class DatasetProcessor_NBA(DatasetProcessor_BASE):
    def __init__(self, args):
        assert  self.args.dataset == "NBA" or self.args.dataset == "Soccer"
        print(self.args.dataset)
        self.args = args
        print("现在真实执行的数据集是" + self.args.dataset)
        """
        如果 self.args.model 不等于 "new_star_hin"，则 assert 语句通过，不进行任何操作。
        如果 self.args.model 等于 "new_star_hin"，则检查 self.args.HIN 是否为 True。
        如果 self.args.HIN 为 True，则 assert 语句通过；如果不为 True，则引发 AssertionError 并显示指定的错误消息。
        """
        assert self.args.train_model != "new_star_hin" or self.args.HIN, "当 model 为 'new_star_hin' 时，HIN 必须为 True"
        self.test_sne_cache = os.path.join(self.args.save_dir, 'test_sne_cache.cpkl')
        # 需要注意考虑seti-batch是否有存在的必要性
        self.train_seti_batch_cache = os.path.join(self.args.data_dir, "train_seti_batch_cache.cpkl")

        print("数据处理开始 原始数据读取分配工作")
        self.data_preprocess_for_origintrajectory(args=args)
        # ----origin_preprocess --------------------
        print("开始将数据处理成四类形式 （frameped_dict, pedtrajec_dict, scene_list, pedlabel_dict） ")
        # 存于 ./output/eth_ucy/test_set/中即可
        self.train_data_file = os.path.join(self.args.data_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.data_dir, "test_trajectories.cpkl")
        if not (os.path.exists(self.train_data_file) and os.path.exists(self.test_data_file)):
            print("之前并不存在该四类数据 故而重新处理")
            self.data_preprocess_for_transformer(setname='train')
            self.data_preprocess_for_transformer(setname='test')
        print("完成处理，开始将其处理成对应的batch型数据")

        # ---选取测试数据集batch
        # test 在全局通用
        # 存于 ./output/eth_ucy/test_set/中即可
        self.test_batch_cache = os.path.join(self.args.data_dir, "test_batch_cache.cpkl")
        self.test_batch_cache_split = os.path.join(self.args.data_dir, "test_batch_cache_0.cpkl")
        if not (os.path.exists(self.test_batch_cache) or os.path.exists(self.test_batch_cache_split)):
            print("处理对应的test-batch数据")
            self.test_traject_dict  = self.load_dict(self.test_data_file)
            self.data_preprocess_for_originbatch('test')
        self.test_batch, self.test_batchnums = self.load_cache(self.test_batch_cache)
        print('test的数据批次总共有', self.test_batchnums)

        self.train_traject_dict = self.load_dict(self.train_data_file)
        # ---依据不同的情况选择对应的train数据集batch
        if self.args.stage == 'origin' and self.args.phase == 'train':
            print("origin::处理对应的origin-train-batch数据")
            # ----origin_for_transformer----------------
            self.train_origin_batch_cache = os.path.join(self.args.data_dir, "train_origin_batch_cache.cpkl")
            self.train_origin_batch_cache_split = os.path.join(self.args.data_dir, "train_origin_batch_cache_0.cpkl")
            if not (os.path.exists(self.train_origin_batch_cache) or os.path.exists(self.train_origin_batch_cache_split)):
                self.data_preprocess_for_originbatch('train')
            # todo 如果存在相应的split——0——1文件，则需要对应的判断并计算出有多少个这样的文件 不然每次都要处理一次 ！！
            self.train_batch, self.train_batchnums = self.load_cache(self.train_origin_batch_cache)
            print('origin::train的数据批次总共有:', self.train_batchnums)

        elif self.args.stage == 'MLDG' and self.args.phase == 'train':
            print("MLDG::处理对应的MLDG-train-batch数据")
            # 存于 ./output/eth_ucy/test_set/model/
            self.train_MLDG_batch_cache = os.path.join(self.args.data_dir, "train_MLDG_batch_cache.cpkl")
            self.train_MLDG_batch_cache_split = os.path.join(self.args.data_dir, "train_MLDG_batch_cache_0.cpkl")
            if not (os.path.exists(self.train_MLDG_batch_cache) or os.path.exists(self.train_MLDG_batch_cache_split)):
                self.data_preprocess_for_MLDGtask('train')
            self.train_batch_MLDG_task, self.train_batch_MLDG_tasknums = self.load_cache(self.train_MLDG_batch_cache)
            print('MLDG::train的数据批次总共有:', self.train_batch_MLDG_tasknums)

        elif self.args.stage == 'MVDG' and self.args.phase == 'train':
            print("MVDG::处理对应的MVDG-train-batch数据")
            self.train_MVDG_batch_cache = os.path.join(self.args.data_dir, "train_MVDG_batch_cache.cpkl")
            self.train_MVDG_batch_cache_split = os.path.join(self.args.data_dir, "train_MVDG_batch_cache_0.cpkl")
            if not (os.path.exists(self.train_MVDG_batch_cache) or os.path.exists(self.train_MVDG_batch_cache_split)):
                self.data_preprocess_for_MVDGtask('train')
            self.train_batch_MVDG_task, self.train_batch_MVDG_tasknums = self.load_cache(self.train_MVDG_batch_cache)
            print('Total number of training MVDG task batches :', self.train_batch_MVDG_tasknums)

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)

        print("正确完成真实数据集"+self.args.dataset+"的初始化过程")

    # 复用的代码结构
    """
    # def __init__
    # def reset_batch_pointer(self, set, valid=False):

    # def process_data(self, args, data):
    =====数据打包处理保存类
    # def load_dict(self,data_file):
    # def load_cache(self, cachefile):
    # def pick_cache(self):
    =====数据训练过程中获取类
    # def rotate_shift_batch(self, batch_data, ifrotate=True)
    # def get_train_batch(self, idx)
    # def get_test_batch(self, idx)
    =====通用结构类 
    # ！！重写 get_data_index
    # ！！重写 get_data_index_single
    # def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip,return_len)
    =====顶层设计类
    # ！！重写 data_preprocess_for_originbatch(self, setname):
    # def data_preprocess_for_MVDGtask(self,setname):
    # ！！重写 data_preprocess_for_originbatch_split(self):
    =====batch数据形成类
    # def massup_batch(self,batch_data):
    # def get_social_inputs_numpy_HIN(self)
    
    """
    # =====================================================================
    # 下面的方法都必须重新实现------------------------------------------
    def data_preprocess_for_origintrajectory(self, args):
        """
            完成从最原始数据（csv）到初步处理的过程
            完成原traject——preprocess 工作
            该数据 整体只存一份
        ====================================
        1.完成数据获取工作 划分训练 测试 与对应的skip数据
         """
        DATASET_NAME_TO_NUM = {'CLE': 0,'GSW': 1,'NYK': 2,'OKC': 3,'SAS': 4,}
        if args.test_set in ['CLE', 'GSW', 'NYK', 'OKC', 'SAS']:
            self.data_dirs = ['./data/NBA/nba/source/CLE', './data/NBA/nba/source/GSW', './data/NBA/nba/source/NYK',
                              './data/NBA/nba/source/OKC', './data/NBA/nba/source/SAS']
            # Data directory where the pre-processed pickle file resides
            # self.data_dir = './data'
            skip = [10, 10, 10, 10, 10]
            train_set = [i for i in range(len(self.data_dirs))]
            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)
            test_set = DATASET_NAME_TO_NUM[args.test_set]
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
            self.trainskip = [10, 10, 10, 10]
            self.testskip = [10]
        self.args.seq_length = 15
        self.args.obs_length = 5
        self.args.pred_length = 10
        print("完成对NBA数据的 训练测试划分")

    def data_preprocess_for_transformer(self, setname):
        """
        NBA 不太一样 有许多需要重写

        """
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

    def data_preprocess_for_originbatch(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname == 'train':
            # 不一样 没有对应的frame-list与ped-list
            traject_dict = self.train_traject_dict
            cachefile = self.train_batch_cache
            shuffle = True
        else:
            traject_dict = self.test_traject_dict
            cachefile = self.test_batch_cache
            shuffle = False

        data_index = self.get_data_index(traject_dict, setname, ifshuffle=shuffle)
        # traject——dict不一样
        trainbatch = self.get_seq_from_index_balance(traject_dict, data_index, setname)
        trainbatchnums = len(trainbatch)
        print(str(setname) +'NBA batch_nums:' + str(trainbatchnums))
        self.pick_cache(trainbatch=trainbatch, trainbatch_nums=trainbatchnums, cachefile=cachefile)

    def data_preprocess_for_originbatch_split(self):
        # 第一步 加载对应数据集以及相应参数
        traject_dict = self.train_traject_dict
        shuffle = True
        trainbatch_meta = []
        trainbatchnums_meta = []
        # 注意此处的traject-dict为一个dict，dict中包含相应的4个ndarray，对其迭代不能使用enumerate ，该函数适用于迭代列表，元组或其他可迭代对象，字典用items
        # 第二步 按场景分解获取对应batch数据
        for seti,seti_traject_dict in traject_dict.items():
            trainbatch_meta.append({})
            # data-index 处按相应的场景分开即可
            train_index = self.get_data_index_single(seti=seti,data_dict=seti_traject_dict,setname="train",ifshuffle=shuffle)
            train_batch = self.get_seq_from_index_balance(traject_dict=traject_dict,data_index=train_index, setname="train")
            trainbatchnums = len(train_batch)
            # list（场景号） - list（windows号） -tuple （）
            trainbatch_meta[seti] = train_batch
            trainbatchnums_meta.append(trainbatchnums)
        self.trainbatch_meta = trainbatch_meta
        self.trainbatchnums_meta = trainbatchnums_meta

    def data_preprocess_for_MLDGtask(self, setname):
        """
             基于data_preprocess_transformer将数据处理成MLDG可以运用的task结构类型
             完成原meta——task工作
         """
        # 第一步 加载对应数据集以及相应参数
        # 第二步 按场景分解获取对应batch数据
        self.data_preprocess_for_originbatch_split()
        # 第三步 形成task-list
        trainbatch_meta = self.trainbatch_meta
        trainbatchnums_meta = self.trainbatchnums_meta
        cachefile = self.train_MLDG_batch_cache
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
        batch_task_list = [task_list[i:i + self.args.query_sample_num] for i in
                      range(0, len(task_list), self.args.query_sample_num)]
        batch_task_num = len(batch_task_list)
        self.pick_cache(trainbatch=batch_task_list, trainbatch_nums=batch_task_num, cachefile=cachefile)

    # =======实际具体任务

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

    def get_data_index_single(self, seti, data_dict, setname, ifshuffle=True):
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

    def get_seq_from_index_balance(self, traject_dict, data_index, setname):
        if self.args.HIN:
            print("MLDG任务中生成的数据是基于HIN的")
            batch = self.get_seq_from_index_balance_HIN(traject_dict=traject_dict, data_index=data_index, setname=setname)
        else:
            print("MLDG任务中生成的数据是基于同质图的")
            batch = self.get_seq_from_index_balance_origin(traject_dict=traject_dict, data_index=data_index, setname=setname)
        return batch

    def get_seq_from_index_balance_HIN(self, traject_dict, data_index, setname):
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
        # 完整的15帧数据以及处理好的 故而后续不在需要进行降采样
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

            cur_trajec = cur_trajec.transpose(1, 0, 2)
            # 前述步骤不一样 主要是因为相对的人数是固定的 故而简化 目标都是获取相对的 frmea-list，ped-list，2 格式数据
            batch_pednum = sum([i.shape[1] for i in batch_data]) + cur_trajec.shape[1]

            cur_pednum = cur_trajec.shape[1]
            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame,)

            batch_data.append(cur_trajec)
            Batch_id.append(batch_id)

            if batch_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                batch_data = self.massup_batch_HIN(batch_data=batch_data,setname=setname)
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
            batch_data = self.massup_batch_HIN(batch_data=batch_data,setname=setname)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数值')
            # batch_data_mass.append((batch_data, Batch_id,))
        return batch_data_mass

    def massup_batch_HIN(self, batch_data,setname):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        if self.args.dataset == "ETH_UCY":
            relation_num = 1
        elif self.args.dataset == "SDD" or self.args.dataset == "NBA" or self.args.dataset == "soccer":
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

    def get_social_inputs_numpy_HIN(self, inputnodes, relation_num,setname):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]
        relation_matrix = np.zeros((num_Peds, num_Peds))
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
        nei_list = np.zeros((relation_num, inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((relation_num, inputnodes.shape[0], num_Peds))

        for relation in range(relation_num):
            # nei_list[f,i,j] denote if j is i's neighbors in frame f
            cur_nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
            cur_nei_num = np.zeros((inputnodes.shape[0], num_Peds))
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
                            abs(relative_cord[:, 1]) > self.args.neighbor_thred) | (
                                              relation_matrix[pedi][pedj] != relation and pedi != pedj)
                    # 自己与自己需要删除

                    cur_nei_num[select, pedi] -= select_dist

                    select[select == True] = select_dist
                    cur_nei_list[select, pedi, pedj] = 0
            nei_list[relation, :, :, :] = cur_nei_list
            nei_num[relation, :, :] = cur_nei_num

        return seq_list, nei_list, nei_num

    # 留待后续开发 为NBA设计相关的无HIN结构 代码

    def get_seq_from_index_balance_origin(self, traject_dict, data_index, setname):
        """
        留待后续开发
        """
        # NBA的需要进行验证
        # skip 等其他还是不一样的 需要思考开发一下
        # 虽然相应的massup——batch和get_social_inputs_numpy应该是一样的
        pass



