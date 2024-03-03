import pickle
import random
import numpy as np
import torch
import os
from DataProcessor.SDD_Dataloader import traject_preprocess_SDD

torch.manual_seed(0)
# todo 测试只给了5个，总的数据集有8个，有部分只用作训练？ 命名和实际部分不太符合
DATASET_NAME_TO_NUM = {
    'eth': 0,
    'hotel': 1,
    'zara1': 2,
    'zara2': 3,
    'univ': 4
}


class Trajectory_Dataloader():
    def __init__(self, args):

        self.args = args
        # 将self.args.save_dir 全部换成self.args.model_dir 则可以拆分开origin-meta-mvdg的数据 互不影响
        # self.log_file_batch_pednum = open(os.path.join(self.args.model_dir, 'meta_batch_pednum.txt'), 'a+')
        self.train_data_file = os.path.join(self.args.model_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.model_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.model_dir, "train_batch_cache.cpkl")
        self.test_batch_cache = os.path.join(self.args.model_dir, "test_batch_cache.cpkl")
        self.test_sne_cache = os.path.join(self.args.model_dir,'test_sne_cache.cpkl')
        # -----meta-----------
        self.train_seti_batch_cache = os.path.join(self.args.model_dir, "train_seti_batch_cache.cpkl")
        self.train_meta_batch_cache = os.path.join(self.args.model_dir, "train_meta_batch_cache.cpkl")
        # -----MVDG-------------
        self.train_MVDG_batch_cache = os.path.join(self.args.model_dir, "train_MVDG_batch_cache.cpkl")
        # todo 数据处理 并不需要每次都从头开始处理 ！
        if self.args.dataset == 'ETH_UCY':
            self.train_dataset = 'ETH_UCY'
            self.test_dataset = 'ETH_UCY'

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
            test_set = DATASET_NAME_TO_NUM[args.test_set]

            if test_set == 4 or test_set == 5:
                self.test_set = [4, 5]
            else:
                self.test_set = [test_set]
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
        elif self.args.dataset == 'SDD':
            self.train_dataset = 'SDD'
            self.test_dataset = 'SDD'
            self.SDD_skip = 12  # 1s30帧，间隔12帧取一注释帧 相应的 skip=1/30*12 = 0.4s 符合8帧预测12帧 3.2s预测4.8s
            print("creating pre_processed data from SDD raw data")
            if not (os.path.exists(self.train_data_file) and os.path.exists(self.test_data_file)):
                traject_preprocess_SDD(test_set=self.args.test_set,train_data_file=self.train_data_file,test_data_file=self.test_data_file)
                # self.traject_preprocess_SDD('train') # self.traject_preprocess_SDD('test')
            print("Done.")
        # todo 此处后续的数据集数据迁移实验  单独在进行分析
        elif self.args.dataset == 'ETH_UCY-SDD':
            self.train_dataset = 'ETH_UCY'
            self.test_dataset = 'SDD'
            self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                              './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                              './data/ucy/univ/students001', './data/ucy/univ/students003',
                              './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']

            # Data directory where the pre-processed pickle file resides
            self.data_dir = './data'
            #  每个场景中注释的帧间隔
            skip = [6, 10, 10, 10, 10, 10, 10, 10]
            train_set = [i for i in range(len(self.data_dirs))]
            # 获取对应的train和test数据集的地址以及skip 后续应该需要添加对应的val，也可以从train中抽取一部分作为val
            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.trainskip = [skip[x] for x in train_set]
            # todo 注意后续添加文件夹区别出与现有的区别？？ETH完整的行人数据作为train
            self.SDD_skip = 12
            if not (os.path.exists(self.train_data_file) and os.path.exists(self.test_data_file)):
                print('将ETH-UCY的完整数据作为训练集')
                self.traject_preprocess('train')
                # SDD的测试集数据
                print('将SDD的部分数据作为测试集')
                self.traject_preprocess_SDD('test')
            print('Done')
        elif self.args.dataset == 'SDD-ETH_UCY':
            self.train_dataset = 'SDD'
            self.test_dataset = 'ETH_UCY'
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
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.testskip = [skip[x] for x in self.test_set]
            self.SDD_skip = 12
            if not (os.path.exists(self.train_data_file) and os.path.exists(self.test_data_file)):
                print('将ETH-UCY的部分数据作为测试集')
                self.traject_preprocess('test')
                print('将SDD的部分数据作为训练集')
                self.traject_preprocess_SDD('train')
            print('Done')
        else:
            raise NotImplementedError
        # 处理数据保存的地址 test
        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict,self.test_scene_list = self.load_dict(self.test_data_file)
            self.dataPreprocess('test',dataset=self.test_dataset)
        self.testbatch, self.test_batchnums, _, _ = self.load_cache(self.test_batch_cache)

        print('Total number of test batches:', self.test_batchnums)
        # 依据条件处理训练集数据
        if self.args.stage == 'origin':
            # Load the processed origin data from the pickle file （原始非meta）
            print("Preparing origin data batches.")
            if not (os.path.exists(self.train_batch_cache)):
                self.frameped_dict, self.pedtraject_dict,self.train_scene_list = self.load_dict(self.train_data_file)
                self.dataPreprocess(setname ='train',dataset=self.train_dataset)
            # 为对比实验而准备 todo pickle data was truncated // run out of input
            self.trainbatch, self.train_batchnums, _, _ = self.load_cache(self.train_batch_cache)
            print('Total number of training batches:', self.train_batchnums)

        elif self.args.stage == 'meta':
            # Load the meta-processed data from the pickle file
            print("Preparing seti data batches.")
            if not (os.path.exists(self.train_seti_batch_cache)):
                self.frameped_dict, self.pedtraject_dict,self.train_scene_list = self.load_dict(self.train_data_file)
                self.dataPreprocess_meta('train',dataset=self.train_dataset)
            print("Preparing meta task data batches.")
            if not (os.path.exists(self.train_meta_batch_cache)):
                print("process train meta cpkl")
                self.batchdata_meta, self.batchnums_meta,self.train_scene_list, _, _ = self.load_cache(self.train_seti_batch_cache)
                self.meta_task(setname="train",dataset=self.train_dataset)
            self.train_batch_MLDG_task = self.load_cache(self.train_meta_batch_cache)
            print('Total number of training meta task batches :', len(self.train_batch_MLDG_task))

        elif self.args.stage == 'MVDG' or self.args.stage =='MVDGMLDG':
            print("Preparing seti data batches.")
            if not (os.path.exists(self.train_seti_batch_cache)):
                self.frameped_dict, self.pedtraject_dict,self.train_scene_list = self.load_dict(self.train_data_file)
                self.dataPreprocess_meta('train', dataset=self.train_dataset)
            print("Preparing MVDG task data batches.")
            if not (os.path.exists(self.train_MVDG_batch_cache)):
                print("process train MVDG cpkl")
                self.batchdata_MVDG, self.batchnums_MVDG,self.train_scene_list, _, _ = self.load_cache(self.train_seti_batch_cache)
                # 注意shuffle！！--经过试验后发现shuffle的效果更差
                self.MVDG_task(setname="train",dataset =self.train_dataset,ifshuffle1=False)
            # 注意多层封装 每批 3个轨迹 每个轨迹内部4个task 每个task对应包含一个train和test 每个train包含256个行人
            self.train_batch_MVDG_task = self.load_cache(self.train_MVDG_batch_cache)

        # 单独分析 test-sne:
        if args.phase == 'test' and args.vis == 'sne':
            self.test_snedata = self.dataPreprocess_sne()

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)


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
        all_frame_data = []
        valid_frame_data = []
        numFrame_data = []
        Pedlist_data = []

        frameped_dict = []  # peds id contained in a certain frame
        pedtrajec_dict = []  # trajectories of a certain ped
        scene_list = []
        # For each dataset
        for seti, directory in enumerate(data_dirs):
            # 4 （frame，Ped-ID y x）
            scene_id = directory.split('/')[-1]
            scene_list.append(scene_id)
            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset
            # 获取当前数据集中所有行人的 ID
            Pedlist = np.unique(data[1, :]).tolist()
            numPeds = len(Pedlist)
            # Add the list of frameIDs to the frameList_data
            Pedlist_data.append(Pedlist)
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])
            # 整个数据集
            numFrame_data.append([])
            # 记录了当前数据集的每个帧包含了那些行人
            frameped_dict.append({})
            # 记录了每个行人的轨迹数据 （数据集，行人id，该行人的帧，对应帧下的xy数据）
            pedtrajec_dict.append({})

            for ind, pedi in enumerate(Pedlist):
                if ind % 100 == 0:
                    print(ind, len(Pedlist))
                # Extract trajectories of one person 抽取单人的轨迹数据
                FrameContainPed = data[:, data[1, :] == pedi]
                # Extract peds list
                FrameList = FrameContainPed[0, :].tolist()
                if len(FrameList) < 2:
                    continue
                # Add number of frames of this trajectory
                numFrame_data[seti].append(len(FrameList))
                # Initialize the row of the numpy array
                Trajectories = []
                # For each ped in the current frame

                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    # todo 按列筛选 后续也可以基于其运行的结果将其按数据集分开
                    current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                    current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                    # Add their pedID, x, y to the row of the numpy array
                    Trajectories.append([int(frame), current_x, current_y])
                    # 如果当前帧不在frameped_dict中，则相应的添加该帧，并将该帧包含的行人添加；记录了当前数据集的每个帧包含了那些行人
                    if int(frame) not in frameped_dict[seti]:
                        frameped_dict[seti][int(frame)] = []
                    frameped_dict[seti][int(frame)].append(pedi)
                pedtrajec_dict[seti][pedi] = np.array(Trajectories)
        # open 函数以二进制写入模式打开指定的文件 data_file，返回一个文件对象 f
        f = open(data_file, "wb")
        # 这两个对象序列化到文件中
        pickle.dump((frameped_dict, pedtrajec_dict,scene_list), f, protocol=2)
        f.close()

    def dataPreprocess(self, setname,dataset):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname == 'train':
            # todo val 为  0 ？
            val_fraction = 0
            frameped_dict = self.frameped_dict
            pedtraject_dict = self.pedtraject_dict
            cachefile = self.train_batch_cache

        else:
            val_fraction = 0
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            cachefile = self.test_batch_cache
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True

        print(setname)

        data_index = self.get_data_index(frameped_dict, setname, ifshuffle=shuffle)
        # data-index:各自数据集中所有帧 ID 和它们所属的数据集 ID、数字化后（打乱的）的帧 ID 存储在一个 3 x N 的 NumPy 数组
        val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        train_index = data_index[:, int(data_index.shape[1] * val_fraction):]
        # todo 依行人总数累加windows 改成依数据集累加
        trainbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, train_index, setname,dataset)
        valbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, val_index, setname,dataset)
        trainbatchnums = len(trainbatch)
        valbatchnums = len(valbatch)

        f = open(cachefile, "wb")
        pickle.dump((trainbatch, trainbatchnums, valbatch, valbatchnums), f, protocol=2)
        f.close()

    def dataPreprocess_meta(self, setname,dataset):
        """
        拆分各自数据集形成得到各自的windows窗口，打包形成各自的batch
        组合成task
        """

        if setname == 'train':
            val_fraction = 0
            frameped_dict = self.frameped_dict
            pedtraject_dict = self.pedtraject_dict
            scene_list = self.train_scene_list
            cachefile = self.train_seti_batch_cache

        else:
            # todo 'Trajectory_Dataloader' object has no attribute 'test_frameped_dict'
            val_fraction = 0
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            scene_list =self.test_scene_list
            cachefile = self.test_seti_batch_cache
        if setname != 'train':
            # 数据集内部各自shuffle
            shuffle = False
        else:
            shuffle = True
        # 运用for循环，分别处理单个数据集
        trainbatch_meta = []
        valbatch_meta = []
        trainbatchnums_meta = []
        valbatchnums_meta = []
        # todo 其实拆不拆分 数据的量是不变的，但是随机性可能会少一点，可能出现在bookstore采样support，仍然在bookstore采样query的情况 可以考虑在后期进行调节 即task组合的地方 选取时更改一下即可
        #  此处针对于SDD数据集，可以把同一场景不同视频汇总到一起  注意此处混合到一起后 时序关系不能混乱 混合较为麻烦 不同数据下的行人ID应该是不一样的，但这里混合成一样的了
        # train: bookstore0123[00-03] coupa3[04] deathCircle01234[05-09] gate01345678[10-17] hyang45679[18-22] nexus0134789[23-29]
        # test: coupa01[00-01] gates2[02] hyang0138[03-06] little0123[07-10] nexus56[11-12] quad0123[13-16]
        new_frameped_dict = frameped_dict
        new_pedtraject_dict = pedtraject_dict
        # [13, 6, 5, 4, 8, 10, 13, 0, 6, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 11, 0, 1, 3, 4, 2]
        # [14, 7, 6, 5, 10, 10, 13, 1, 7, 1, 1, 3, 5, 1, 1, 1, 1, 1, 1, 5, 5, 1, 0, 1, 11, 1, 2, 4, 5, 3]
        # 8-10 多加了一个 有空白？？coupa   ?
        for seti, seti_frameped_dict in enumerate(new_frameped_dict):
            # 提取出单个数据集
            trainbatch_meta.append({})
            valbatch_meta.append({})
            data_index = self.get_data_index_meta(seti, seti_frameped_dict, setname, ifshuffle=shuffle)
            val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
            train_index = data_index[:, int(data_index.shape[1] * val_fraction):]
            trainbatch = self.get_seq_from_index_balance_meta(seti, seti_frameped_dict, new_pedtraject_dict,
                                                              train_index, setname,dataset)
            valbatch = self.get_seq_from_index_balance_meta(seti, seti_frameped_dict, new_pedtraject_dict, val_index,
                                                            setname,dataset)
            trainbatchnums = len(trainbatch)
            valbatchnums = len(valbatch)
            # list（场景号） - list（windows号） -tuple （）
            trainbatch_meta[seti] = trainbatch
            valbatch_meta[seti] = valbatch
            trainbatchnums_meta.append(trainbatchnums)
            valbatchnums_meta.append(valbatchnums)
        # 动态的迭代选取support和query组成task格式为（task-num，2）（0-support，1-query）
        # todo 分析相应的结果值
        f = open(cachefile, "wb")
        pickle.dump((trainbatch_meta, trainbatchnums_meta,scene_list,valbatch_meta, valbatchnums_meta), f, protocol=2)
        f.close()

    def dataPreprocess_sne(self):
        # 按场景提取混合数据存在字典data_dict中
        # 直接调用test-batch-cache和meta的train—batch-cache
        # 获取test的数据
        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict, self.test_scene_list = self.load_dict(self.test_data_file)
            self.dataPreprocess('test', dataset=self.test_dataset)
        self.test_batch, self.test_batchnums, _, _ = self.load_cache(self.test_batch_cache)
        # 获取train的数据
        if not (os.path.exists(self.train_seti_batch_cache)):
            self.frameped_dict, self.pedtraject_dict, self.train_scene_list = self.load_dict(self.train_data_file)
            self.dataPreprocess_meta('train', dataset=self.train_dataset)
        self.batchdata_meta, self.batchnums_meta, self.train_scene_list, _, _ = self.load_cache(
                self.train_seti_batch_cache)
        # 合并test和train的数据
        data_dict = {'hotel':[],'zara01':[],'zara02':[],'univ':[],'eth':[]}
        # 对应关系 eth-univ，hotel，zara01，zara02，univ-[students001,students002]
        data_dict[self.args.test_set].extend(self.test_batch)
        for id,scene in enumerate(self.train_scene_list):
            if scene in['hotel','zara01','zara02']:
                data_dict[scene].extend(self.batchdata_meta[id])
            elif scene == 'univ':
                data_dict['eth'].extend(self.batchdata_meta[id])
            elif scene == 'students001' or scene == 'students003':
                data_dict['univ'].extend(self.batchdata_meta[id])
        return data_dict

    def meta_task(self, setname, dataset):
        """
        1-组合各个数据集的batch数据，循环遍历数据集0-6，针对每个数据集中的每个batch，重复四次选取，support一样，query从其他场景中随机挑选，
        两次随机，随机选数据集号，而后再随机选数据集号下对应的batch，从而组合成一个tuple。
        2-反复如此操作，得到最终的task列表
        3-打乱task列表后，依顺序4个组，组成batch。

        todo 后续考虑：
        1-此处划分的数据集，有多个可能是同属于一个场景，在这先认为一样，后续引入相应的场景序号，场景序号下视频序号
        2-task池做大，相应的阈值可设256，不影响。
        3-针对test？再说吧,如何形成batch？
        """
        if setname == 'train':

            cachefile = self.train_meta_batch_cache
            task_list = []
            for seti, seti_batch_num in enumerate(self.batchnums_meta):
                # 此处会有不同 ETH-UCY数据集直接移除相应的i即可 但是针对于SDD数据集,需要先聚合相同场景的代码
                if self.batchnums_meta[seti] == 0 or self.batchnums_meta[seti] ==[]:
                    continue
                query_seti_id = list(range(len(self.batchnums_meta)))
                if dataset == 'ETH_UCY':
                    query_seti_id.remove(seti)
                elif dataset == 'SDD':
                    # 第一步依据seti以及对应的scene-list找出与set相同的场景，其他不同的加入到query——seti-id里
                    scene_now = self.train_scene_list[seti]
                    # 从字符串"bookstore_0"中提取出"bookstore"
                    scene_now = scene_now[:-2]
                    for i in range(len(self.train_scene_list)):
                        scene_find = self.train_scene_list[i][:-2]
                        if scene_find == scene_now:
                            query_seti_id.remove(i)
                for batch_id in range(seti_batch_num):
                    support_set = self.batchdata_meta[seti][batch_id]
                    # support-set 为tuple 包含tuple和list，tuple中又有0,1,2,3个ndarray和1个list
                    if len(support_set[0][0]) == 0 or len(support_set) == 0:  # 需要深入分析
                        continue
                    for query_i in range(self.args.query_sample_num):
                        random_query_seti = random.choice(query_seti_id)
                        while len(self.batchdata_meta[random_query_seti][0][0]) == 0 or len(self.batchdata_meta[random_query_seti])==0:
                            random_query_seti = random.choice(query_seti_id)
                        random_query_seti_batch = random.randint(0, self.batchnums_meta[random_query_seti] - 1)
                        query_set = self.batchdata_meta[random_query_seti][random_query_seti_batch]
                        task_list.append((support_set, query_set,))
            # todo 最开始是按顺序获取task，获取完毕后打乱task  针对于sequential 不太适合 因为batch中的4个query task不一样，无法充分利用query数据
            # random.shuffle(task_list)
            batch_task = [task_list[i:i + 4] for i in range(0, len(task_list), 4)]
            print("Finsh task batch" + str(setname))
        else:
            # todo 有待下一步开发
            self.batchdata_meta, self.batchnums_meta, _, _ = self.load_cache(self.test_seti_batch_cache)
            cachefile = self.test_meta_batch_cache

        f = open(cachefile, "wb")
        pickle.dump(batch_task, f, protocol=2)
        f.close()

    def MVDG_task(self, setname, dataset, ifshuffle1=False):
        """
        1-组合各个数据集的batch数据，循环遍历数据集0-6，针对每个数据集中的每个batch，重复n次选取，support一样，query从其他场景中随机挑选，
        两次随机，随机选数据集号，而后再随机选数据集号下对应的batch，从而组合成一个tuple。
        2-反复如此操作，得到最终的task列表
        3-不打乱task列表后，依顺序4个组，组成batch。

        todo 后续考虑：
        1-此处划分的数据集，有多个可能是同属于一个场景，在这先认为一样，后续引入相应的场景序号，场景序号下视频序号 -》针对于SDD的已经修改
        2-task池做大，相应的阈值可设256-》512，不影响。
        3-针对test？再说吧,如何形成batch？
        """
        if setname == 'train':
            cachefile = self.train_MVDG_batch_cache
            task_list = []
            for seti, seti_batch_num in enumerate(self.batchnums_MVDG):
                # 此处会有不同 ETH-UCY数据集直接移除相应的i即可 但是针对于SDD数据集,需要先聚合相同场景的代码
                if self.batchnums_MVDG[seti] == 0 or self.batchnums_MVDG[seti] == []:
                    continue
                query_seti_id = list(range(len(self.batchnums_MVDG)))
                if dataset == 'ETH_UCY':
                    query_seti_id.remove(seti)
                elif dataset == 'SDD':
                    # 第一步依据seti以及对应的scene-list找出与set相同的场景，其他不同的加入到query——seti-id里
                    scene_now = self.train_scene_list[seti]
                    # 从字符串"bookstore_0"中提取出"bookstore"
                    scene_now = scene_now[:-2]
                    for i in range(len(self.train_scene_list)):
                        scene_find = self.train_scene_list[i][:-2]
                        if scene_find == scene_now:
                            query_seti_id.remove(i)
                for batch_id in range(seti_batch_num):
                    support_set = self.batchdata_MVDG[seti][batch_id]
                    if len(support_set[0][0]) == 0 or len(support_set) == 0:
                        continue
                    for query_i in range(self.args.query_sample_num):
                        random_query_seti = random.choice(query_seti_id)
                        while len(self.batchdata_MVDG[random_query_seti][0][0]) == 0 or len(self.batchdata_MVDG[random_query_seti]) == 0:
                            random_query_seti = random.choice(query_seti_id)
                        random_query_seti_batch = random.randint(0, self.batchnums_MVDG[random_query_seti] - 1)
                        query_set = self.batchdata_MVDG[random_query_seti][random_query_seti_batch]
                        task_list.append((support_set, query_set,))
            # 最开始是按顺序获取task，获取完毕后打乱task todo mvdg同一优化轨迹下不需要打乱 现在的写法
            #  todo (其实也需要打乱，按照描写的算法而言，应该是每个task都重新采样训练和测试，此处简化了！！) 但不打乱效果更好
            if ifshuffle1:
                random.shuffle(task_list)
            # todo 此处的4需要改成self.args.query_sample_num
            batch_task_list = [task_list[i:i + self.args.query_sample_num] for i in
                               range(0, len(task_list), self.args.query_sample_num)]
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
        else:
            # todo 有待下一步开发
            self.batchdata_MVDG, self.batchnums_MVDG, _, _ = self.load_cache(self.test_seti_batch_cache)
            cachefile = self.test_MVDG_batch_cache

        f = open(cachefile, "wb")
        pickle.dump(new_batch_task_list, f, protocol=2)
        f.close()

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]
        scene_list = raw_data[2]

        return frameped_dict, pedtraject_dict,scene_list

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def get_data_index(self, data_dict, setname, ifshuffle=True):
        '''
        Get the dataset sampling index.
        data-dict：集合，包含了多个场景，每个场景是一个时间序列，其存储了每个场景从第一帧到最后一帧（固定间隔）的行人标号
        setname：train/test
        其主要作用是返回：所有帧 ID 和它们所属的数据集 ID、数字化后的帧 ID 存储在一个 3 x N 的 NumPy 数组 data_index 中
        '''
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        for seti, dict in enumerate(data_dict):
            frames = sorted(dict)
            # 此处maxframe如果是为了使得每个帧都存在未来的20帧，则相应的应该乘以间隔 即 self.args.seq_length*self.skip[seti]
            maxframe = max(frames) - self.args.seq_length
            # 去掉不足一个序列长度的帧 ID
            frames = [x for x in frames if not x > maxframe]
            total_frame += len(frames)
            # 添加一行 标记每个数据所属的数据集
            set_id.extend(list(seti for i in range(len(frames))))
            frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))

        all_frame_id_list = list(i for i in range(total_frame))
        # data-index 格式 （各自数据集中的实际帧号，所属数据集编号，全局数字化后值）
        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        # todo 对于训练集 会打乱data-index 后续可以考虑不打乱 生成task需要对应数据集
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # to make full use of the data
        # ，函数将 data_index 数组的前 batch_size 列复制到数组的末尾，以增加训练集的大小
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
        frames = sorted(data_dict)
        # 此处maxframe如果是为了使得每个帧都存在未来的20帧，则相应的应该乘以间隔 即 self.args.seq_length*self.skip[seti]
        maxframe = max(frames) - self.args.seq_length
        # 去掉不足一个序列长度的帧 ID
        frames = [x for x in frames if not x > maxframe]
        total_frame += len(frames)
        # 添加一行 标记每个数据所属的数据集
        set_id.extend(list(seti for i in range(len(frames))))
        frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))
        all_frame_id_list = list(i for i in range(total_frame))
        # data-index 格式 （各自数据集中的实际帧号，所属数据集编号，全局数字化后值）
        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]
        # to make full use of the data
        # ，函数将 data_index 数组的前 batch_size 列复制到数组的末尾，以增加训练集的大小
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def get_seq_from_index_balance_meta(self, seti, seti_frameped_dict, pedtraject_dict, data_index, setname,dataset):
        """
        各个数据集自身形成batch！ 似乎并没有存在的必要 ！！
        """
        batch_data_mass = []
        batch_data = []
        Batch_id = []
        temp = self.args.batch_around_ped_meta

        if dataset == 'ETH_UCY':
            if setname == 'train':
                skip = self.trainskip
            else:
                skip = self.testskip
        elif dataset == 'SDD':
            skip = self.SDD_skip

        ped_cnt = 0
        last_frame = 0
        # 全局处理 混合所有train的帧 形成的windows
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            # todo 针对于sdd数据存在错误
            framestart_pedi = set(seti_frameped_dict[cur_frame])
            # 计算并获取对应起始帧（子轨迹）的结束帧，由于当前的子轨迹的结束帧可能会超过数据集的范围，因此使用try-expect语句块处理这种情况
            try:
                if dataset == 'ETH_UCY':
                    frameend_pedi = set(seti_frameped_dict[cur_frame + (self.args.seq_length-1) * skip[cur_set]])
                elif dataset == 'SDD':
                    frameend_pedi = set(seti_frameped_dict[cur_frame + (self.args.seq_length-1) * skip])
            except:
                continue

            present_pedi = framestart_pedi | frameend_pedi
            # 如果起始帧与结束帧没有重复的行人id，则抛弃该子轨迹
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            """
            针对由起始帧和结束帧确定的窗口序列以及行人并集，遍历行人，找到该行人在起始帧与结束帧之间存在的片段；若正好全程存在，则iffull为true，
            若有空缺，则iffull为False；ifexistobs标识obs帧是否存在，并删去太短的片段（小于5）；而后去除帧号，只保留这些行人的xy坐标；添加到traject中
            而后将滤除后的行人轨迹数据保留并拼接；batch-pednum为相应的不断累计不同时间窗口轨迹数据的总值，
            """
            for ped in present_pedi:
                # cur-trajec：该行人对应的子轨迹数据（可能是完整的20，也可能小于20） iffull指示其是否满，ifexistobs指示其是否存在我们要求的观测帧
                if dataset == 'ETH_UCY':
                    cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                                   cur_frame,
                                                                                   self.args.seq_length, skip[cur_set])
                elif dataset == 'SDD':
                    cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                                   cur_frame, self.args.seq_length,
                                                                                   skip)
                # 对于每个子轨迹，如果它的长度小于阈值或者该子轨迹中的一些帧没有数据，则忽略该子轨迹；否则加进cur_trajec
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    #  Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                    # 自取cur-trajec的后两列（xy）组成（20,1,2）的向量，此处不直接用（20，2）是为了后续在第二维拼接不同的行人数据
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            # 按照第二个维度进行拼接，即将同一个windows中行人数据拼接在一起
            traject_batch = np.concatenate(traject, 1)
            # 基于后续叠加各个windows中的行人数据
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]
            # 该windows中的行人数量
            cur_pednum = traject_batch.shape[1]
            print(dataset+'_'+setname+'_'+str(cur_pednum))

            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame,)
            #  todo 还未测试
            """
            如果以当前数据集以及相应的预测帧起始的窗口中包含超过512个行人的轨迹，则将其进行拆分为两个batch，如果处于256和512之间，
            将其打包成为一个batch；如果小于256，则相应的累加其他时间窗口的轨迹数据，直到batch里的行人数大于256,将其打包为一个batch             
            """
            # todo 分数据集提取保存！！[set,batch-data]
            """测试完成 关闭
            if i % 10 == 0:
                self.log_file_batch_pednum.close()
                self.log_file_batch_pednum = open(os.path.join(self.args.model_dir, 'meta_batch_pednum.txt'), 'a+')
            """

            if cur_pednum >= self.args.batch_around_ped_meta * 2:
                # too many people in current scene
                # split the scene into two batches
                # self.log_file_batch_pednum.write(str(seti) + '----' + str(cur_pednum) + '\n')
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_batch_data, cur_Batch_id = [], []
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]],
                              traject_batch[:, ind[cur_pednum // 2:, 0]]]
                for sb in Seq_batchs:
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_batch_data = self.massup_batch(cur_batch_data)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                    cur_batch_data = []
                    cur_Batch_id = []

                last_frame = i
            elif cur_pednum >= self.args.batch_around_ped_meta:
                # self.log_file_batch_pednum.write(str(seti) + '----' + str(cur_pednum) + '\n')
                # good pedestrian numbers
                cur_batch_data, cur_Batch_id = [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_batch_data = self.massup_batch(cur_batch_data)
                batch_data_mass.append((cur_batch_data, cur_Batch_id,))

                last_frame = i
            else:  # less pedestrian numbers <64
                # accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped_meta:
                    # self.log_file_batch_pednum.write(str(seti) + '----' + str(batch_pednum) +  '\n')
                    # enough people in the scene
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
                    """
                    输入：多个windows的数据 （windows-num，20，windows-ped，2）
                    batch_data_mass：多个（batch_data, Batch_id）
                    batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

                    nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                    seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                    nei_list_b：(seq_length, num_Peds，num_Peds) (20,257，257）值为01 以空间距离为基准 分析邻接关系
                    nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                    batch_pednum：list 表示该batch下每个时间窗口中的行人数量
                    """
                    batch_data = self.massup_batch(batch_data)
                    if batch_data[4] != []: # 即batch_pednum为空 则不添加该数据
                        batch_data_mass.append((batch_data, Batch_id,))
                    elif batch_data[4] == []:
                        print('舍弃该数值')
                    # batch_data_mass.append((batch_data, Batch_id,))

                    last_frame = i
                    batch_data = []
                    Batch_id = []
                else:
                    # todo batch问题 缺失轨迹预测问题 meta-learning task设计问题
                    """
                     一般都要经过累加，相应的过往的batch处理是选择固定的多个20s的场景数据，而此处区别则是每个batch中包含的20s场景数是不同的
                     其以该batch中的人（轨迹）数量为准，直到累加超过阈值；（好处可能是解决了单个轨迹处理耗时慢的问题，同样解决了部分batch数据轨迹太少）
                     需要注意的是 在meta原本思路中，batch的定义与此处不同，batch以task为基础，每个task只需要一条support和一条query，其皆由单个20s场景组成
                     但很可能存在，即相应的20s内无轨迹，或则只有几条行人轨迹可用；故而可以先进行叠加以64-128个行人轨迹数先组成batch，以此batch为support和query                     
                     （问题二 源代码对应的batch的traj中部分行人数据未全程存在，该部分如何预测）   
                    """
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
        # todo 需要分析针对于不足batch_pednum的最后几个windows的情况，如果是train，则直接舍弃最后的数据，如果是test，而且相应的不是最后一帧，
        #  即没有处理完，而且batch-pednum中行人数大于1，则对其进行相应的batch处理，加到数据集中
        #  if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
        #  需要注意的是 train中的数据也不能直接舍弃，当batch较大的时候，相应的由很多数据形不成512，则会被抛弃，造成数据量的不足 ！！
        if last_frame < data_index.shape[1] - 1 and batch_pednum > 1:
            # self.log_file_batch_pednum.write(str(seti) + '----' + str(batch_pednum) + '\n')
            batch_data = self.massup_batch(batch_data)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数值')
            # batch_data_mass.append((batch_data, Batch_id,))
        self.args.batch_around_ped = temp
        return batch_data_mass

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname,dataset):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
                    batch_data_mass：多个（batch_data, Batch_id）
                    batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
                    nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                    seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                    nei_list_b：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
                    nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                    batch_pednum：list 表示该batch下每个时间窗口中的行人数量
        '''
        batch_data_mass = []
        batch_data = []
        Batch_id = []

        temp = self.args.batch_around_ped
        if dataset == 'ETH_UCY':
            if setname == 'train':
                skip = self.trainskip
            else:
                skip = self.testskip
        elif dataset == 'SDD':
            skip = self.SDD_skip

        ped_cnt = 0
        last_frame = 0
        # 全局处理 混合所有train的帧 形成的windows
        for i in range(data_index.shape[1]):
            '''
            仍然是以对应窗口序列划分 例如test有1443帧，则相应的可以划分处1443个时间窗口，但需要后期依据
            '''
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            # 计算并获取对应起始帧（子轨迹）的结束帧，由于当前的子轨迹的结束帧可能会超过数据集的范围，因此使用try-expect语句块处理这种情况
            try:
                if dataset == 'ETH_UCY':
                    frameend_pedi = set(frameped_dict[cur_set][cur_frame + (self.args.seq_length-1) * skip[cur_set]])
                elif dataset == 'SDD':
                    frameend_pedi = set(frameped_dict[cur_set][cur_frame + (self.args.seq_length-1) * skip])
            except:
                continue
            # todo 合并起始与结束帧中包含的行人
            present_pedi = framestart_pedi | frameend_pedi
            # 如果起始帧与结束帧没有重复的行人id，则抛弃该子轨迹
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            """
            针对由起始帧和结束帧确定的窗口序列以及行人并集，遍历行人，找到该行人在起始帧与结束帧之间存在的片段；若正好全程存在，则iffull为true，
            若有空缺，则iffull为False；ifexistobs标识obs帧是否存在，并删去太短的片段（小于5）；而后去除帧号，只保留这些行人的xy坐标；添加到traject中
            而后将滤除后的行人轨迹数据保留并拼接；batch-pednum为相应的不断累计不同时间窗口轨迹数据的总值，
            """
            for ped in present_pedi:
                # cur-trajec：该行人对应的子轨迹数据（可能是完整的20，也可能小于20） iffull指示其是否满，ifexistobs指示其是否存在我们要求的观测帧
                if dataset == 'ETH_UCY':
                    cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                                   cur_frame,
                                                                                   self.args.seq_length, skip[cur_set])
                elif dataset == 'SDD':
                    cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                                   cur_frame, self.args.seq_length,
                                                                                   skip)
                # 对于每个子轨迹，如果它的长度小于阈值或者该子轨迹中的一些帧没有数据，则忽略该子轨迹；否则加进cur_trajec
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    # todo ？ Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                # 自取cur-trajec的后两列（xy）组成（20,1,2）的向量，此处不直接用（20，2）是为了后续在第二维拼接不同的行人数据
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            # 按照第二个维度进行拼接，即将同一个windows中行人数据拼接在一起
            traject_batch = np.concatenate(traject, 1)
            # 基于后续叠加各个windows中的行人数据
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]
            # 该windows中的行人数量
            cur_pednum = traject_batch.shape[1]
            print(dataset+'_'+setname+'_'+str(cur_pednum))
            ped_cnt += cur_pednum
            # todo 后续基于batch-id进行数据提取
            batch_id = (cur_set, cur_frame,)
            """
            如果以当前数据集以及相应的预测帧起始的窗口中包含超过512个行人的轨迹，则将其进行拆分为两个batch，如果处于256和512之间，
            将其打包成为一个batch；如果小于256，则相应的累加其他时间窗口的轨迹数据，直到batch里的行人数大于256,将其打包为一个batch             
            """
            # todo 分数据集提取保存！！[set,batch-data]
            if cur_pednum >= self.args.batch_around_ped * 2:
                # too many people in current scene
                # split the scene into two batches
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_batch_data, cur_Batch_id = [], []
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]], traject_batch[:, ind[cur_pednum // 2:, 0]]]
                for sb in Seq_batchs:
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_batch_data = self.massup_batch(cur_batch_data)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                    cur_batch_data = []
                    cur_Batch_id = []

                last_frame = i
            elif cur_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                cur_batch_data, cur_Batch_id = [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_batch_data = self.massup_batch(cur_batch_data)
                batch_data_mass.append((cur_batch_data, cur_Batch_id,))

                last_frame = i
            else:  # less pedestrian numbers <64
                # accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped:
                    # enough people in the scene
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
                    """
                    输入：多个windows的数据 （windows-num，20，windows-ped，2）
                    batch_data_mass：多个（batch_data, Batch_id）
                    batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
                    
                    nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                    seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                    nei_list_b：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
                    nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                    batch_pednum：list 表示该batch下每个时间窗口中的行人数量
                    """
                    # todo 需要注意的是后续相应的异质网结构的邻接矩阵会不一样 需要特殊处理 但meatID与label一一对应 可以查询的得到
                    batch_data = self.massup_batch(batch_data)
                    if batch_data[4] != []: # 即batch_pednum为空 则不添加该数据
                        batch_data_mass.append((batch_data, Batch_id,))
                    elif batch_data[4] == []:
                        print('舍弃该数值')
                    last_frame = i
                    batch_data = []
                    Batch_id = []
                else:
                    # todo batch问题 缺失轨迹预测问题 meta-learning task设计问题
                    """
                     一般都要经过累加，相应的过往的batch处理是选择固定的多个20s的场景数据，而此处区别则是每个batch中包含的20s场景数是不同的
                     其以该batch中的人（轨迹）数量为准，直到累加超过阈值；（好处可能是解决了单个轨迹处理耗时慢的问题，同样解决了部分batch数据轨迹太少）
                     需要注意的是 在meta原本思路中，batch的定义与此处不同，batch以task为基础，每个task只需要一条support和一条query，其皆由单个20s场景组成
                     但很可能存在，即相应的20s内无轨迹，或则只有几条行人轨迹可用；故而可以先进行叠加以64-128个行人轨迹数先组成batch，以此batch为support和query                     
                     （问题二 源代码对应的batch的traj中部分行人数据未全程存在，该部分如何预测）   
                    """
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
        # todo 需要分析针对于不足batch_pednum的最后几个windows的情况，如果是train，则直接舍弃最后的数据，如果是test，而且相应的不是最后一帧，
        #  即没有处理完，而且batch-pednum中行人数大于1，则对其进行相应的batch处理，加到数据集中
        #  if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
        #  需要注意的是 train中的数据也不能直接舍弃，当batch较大的时候，相应的由很多数据形不成512，则会被抛弃，造成数据量的不足 ！！
        if last_frame < data_index.shape[1] - 1 and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数据')
            # batch_data_mass.append((batch_data, Batch_id,))
        self.args.batch_around_ped = temp
        return batch_data_mass

    def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip):
        '''
        Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.

        '''
        return_trajec = np.zeros((seq_length, 3))
        # 分析此处是因为【】取不到最后一个，所以多取一个数据
        endframe = startframe + (seq_length) * skip
        start_n = np.where(trajectory[:, 0] == startframe)
        end_n = np.where(trajectory[:, 0] == endframe)
        iffull = False
        ifexsitobs = False
        """
        依据起始帧以及结束帧的存在情况，进行处理，赋予相应的行号，分为四种情况；
        起始帧不存在，结束帧存在：起始帧设为第一帧
        起始帧存在，结束帧不存在：结束帧设为行人轨迹的行数
        起始帧不存在，结束帧不存在：起始帧设为第一帧，并将结束帧设为行人轨迹的行数
        起始帧存在，结束帧存在：设为它们所在行的行号
        """
        if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
            start_n = 0
            end_n = end_n[0][0]
            if end_n == 0:
                return return_trajec, iffull, ifexsitobs

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
            start_n = start_n[0][0]
            end_n = trajectory.shape[0]

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
            start_n = 0
            end_n = trajectory.shape[0]

        else:
            end_n = end_n[0][0]
            start_n = start_n[0][0]

        candidate_seq = trajectory[start_n:end_n]
        # 计算它相对于子轨迹起始帧的偏移量，从而对应的赋予到return-trajec中，如果该行人在这20帧都存在，则返回的轨迹数据也是完整的
        # 如果起始或结束不存在，则相应的return中会有空缺
        offset_start = int((candidate_seq[0, 0] - startframe) // skip)

        offset_end = self.args.seq_length + int((candidate_seq[-1, 0] - endframe) // skip)
        # 无值的地方填0
        # 这样的方法默认其一个人在这个序列里是完整的，中间不会丢帧，如果丢帧了，则candidate_seq拿到的数据会超越20帧，无法往回返。
        # 即SDD需要注意保持轨迹的连续性，如果轨迹在某处分段了，则需要将其拆分
        return_trajec[offset_start:offset_end + 1, :3] = candidate_seq
        # 返回的轨迹长度在观测帧处存在对应的值
        if return_trajec[self.args.obs_length - 1, 1] != 0:
            ifexsitobs = True
        # 返回的轨迹长度大于等于对应要求的序列长度，
        if offset_end - offset_start >= seq_length - 1:
            iffull = True

        return return_trajec, iffull, ifexsitobs

    def massup_batch(self, batch_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]
        # 子轨迹的位置序列 (seq_length, num_Peds) 掩码序列01
        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 2))
        # 邻居列表 (seq_length, num_Peds, num_Peds) 每帧都存在邻接矩阵
        nei_list_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        # 邻居数量  (seq_length, num_Peds) 每帧都存在相应的每个行人的邻居数统计
        nei_num_b = np.zeros((self.args.seq_length, num_Peds))
        # 当前已经处理的行人数量。
        num_Ped_h = 0
        # 存储每个数据片段的行人数量
        batch_pednum = []
        for batch in batch_data:
            # batch-data是一路累加的，相应的每个batch-data中包括的可以除了拼接好的行人轨迹数据外 还可以添加对应的行人的label
            # 作为一个对应的list；
            num_Ped = batch.shape[1]
            # seq-list 为(seq_length, num_Peds) 01值 1表示该行人在该帧有数据
            # nei_list (seq_length, num_Peds, num_Peds)  每帧下的行人邻居关系，基于空间位置计算
            # nei_num (seq_length, num_Peds) 表示每个行人在每帧下的邻居数量
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch)
            # 相应的将该时间窗口的数据添加进batch 按第二维度 即行人的维度 （20，num-ped，2）
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            # 拼接邻接矩阵 互不影响
            nei_list_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            # 指示拼接到何处了
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

    def get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        对于每对行人 (i, j)，计算它们之间的相对坐标，如果相对坐标中任意一个分量的绝对值超过了阈值 self.args.neighbor_thred，则认为它们之间没有邻居关系
        '''
        num_Peds = inputnodes.shape[1]
        # seq-list 表示某一帧下 某个行人是否存在
        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            # 将每个行人每一帧下有数据的标为1
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list inputnodes.shape[0]帧的数量
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            # seq_list中对应的值设置为1，其中数据不缺失（在序列数组的第一列中表示为非零值）。
            # 然后，通过复制seq_list中的值来填充nei_list，并将对角线元（表示当前考虑的行人）设置为0，表示行人不被视为自己的邻居
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                seqj = inputnodes[:, pedj]
                # 选择两个行人都具有非缺失数据的帧
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)
                # 通过从seqi和seqj中减去相应位置的值【：2 即全部数据 xy】，计算两个行人之间的相对坐标。
                relative_cord = seqi[select, :2] - seqj[select, :2]

                # invalid data index 鉴于绝对值的x坐标或y坐标超过阈值（self.args.neighbor_thred），确定无效数据索引。
                select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                        abs(relative_cord[:, 1]) > self.args.neighbor_thred)

                nei_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
                # 主要母的就是填满对应的0-1矩阵
        return seq_list, nei_list, nei_num

    def rotate_shift_batch(self, batch_data, ifrotate=True):
        '''
        Random ration and zero shifting.
        数据旋转增强以及相应的以观测坐标进行坐标的移动归一化
        '''
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data

        # rotate batch
        if ifrotate:
            th = random.random() * np.pi
            cur_ori = batch.copy()
            batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)
        # get shift value 获取位于观测时间最后一帧的位置 s 并重复以使得数据对齐 （seq-length，batch-pednum，2）
        # 以便后续各个帧以该帧为基准进行对齐 坐标归一化操作
        s = batch[self.args.obs_length - 1]

        shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)

        batch_data = batch, batch - shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id = self.train_batch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)

        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.test_batch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False)
        return batch_data, batch_id

    def reset_batch_pointer(self, set, valid=False):
        '''
        Reset all pointers
        set：train/test 表示需要重置的数据集类型 valid 可选参数 表示是否对验证集进行重置
        在每次训练之前将数据集的指针重置，以便下一次训练可以从数据集的第一帧开始处理数据
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0
