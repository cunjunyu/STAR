import os
import numpy as np
import pickle
import random

from DataProcessor.DataProcessorFactory import DatasetProcessor_BASE


class DatasetProcessor_ETH_UCY(DatasetProcessor_BASE):
    def __init__(self, args):
        super().__init__(args=args)
        assert  self.args.dataset == "ETH_UCY"
        print("正确完成真实数据集"+self.args.dataset+"的初始化过程")

    # 复用的代码结构
    """
    # def __init__
    # def reset_batch_pointer(self, set, valid=False):

    =====数据打包处理保存类
    # def load_dict(self,data_file):
    # def load_cache(self, cachefile):
    # def pick_cache(self):
    =====数据训练过程中获取类
    # def rotate_shift_batch(self, batch_data, ifrotate=True)
    # def get_train_batch(self, idx)
    # def get_test_batch(self, idx)
    =====通用结构类 
    # def get_data_index(self, data_dict, setname, ifshuffle=True)
    # def get_data_index_single(self,seti,data_dict, setname, ifshuffle=True):
    # def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip)
    =====通用结构类 
    # def data_preprocess_for_originbatch(self, setname):
    # def data_preprocess_for_MVDGtask(self,setname):
    # def data_preprocess_for_originbatch_split(self):
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

        self.data_dirs = ['./data/eth/univ', './data/eth/hotel', './data/ucy/zara/zara01',
                          './data/ucy/zara/zara02', './data/ucy/univ/students001', './data/ucy/univ/students003',
                          './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']
        skip = [6, 10, 10, 10, 10, 10, 10, 10]
        train_set = [i for i in range(len(self.data_dirs))]
        # 断言 确认相应的test-set在已有数据集中 检查代码中使用的数据集名称是否正确。
        DATASET_NAME_TO_NUM = {'eth': 0, 'hotel': 1, 'zara1': 2, 'zara2': 3, 'univ': 4}
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
        self.args.seq_length = 20
        self.args.obs_length = 8
        self.args.pred_length = 12
        print("完成对eth-ucy数据的 训练测试划分")
        # 处理train和test的数据，相应的得到frameped_dict[每帧包含的行人数]和pedtrajec_dict[每个行人包含的单独轨迹数据]
        # 此处返回的是完整的 8个场景的处理完的所有数据

    def data_preprocess_for_transformer(self, setname):
        """
            将数据处理成后续模型transformer的输入形式
            完成原dataPreprocess  工作 适合于无非区分域或则不需要区分域的形式
            dataPreprocess_sne
            获得 (frameped_dict, pedtrajec_dict, scene_list, pedlabel_dict)
        #=============================================
        完成四类型数据计算与存储：
        智能体的轨迹 pedtrajec_dict 形式pedtrajec_dict[seti][pedi] = np.array(Trajectories);
        每帧包含的智能体标号 frameped_dict;
        智能体的类别 pedlabel_dict ;
        智能体的场景 scene_list;
        尽量精简合并代码 只保留必要的代码 并明确含义
        todo 考虑到后续需要画一个时间窗口内部众多不同人的轨迹数据点
        """
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        elif setname == 'test':
            data_dirs = self.test_dir
            data_file = self.test_data_file
        print("处理__"+setname+"__数据")
        frameped_dict = []   # peds id contained in a certain frame
        pedtrajec_dict = []  # trajectories of a certain ped
        scene_list = []      # 场景名称列表
        pedlabel_dict = []   # label of a certain ped
        # For each dataset
        for seti, directory in enumerate(data_dirs):
            # ----------------读取基础数据
            # 4 （frame，Ped-ID y x）
            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            # -------------- 处理成四个类别的数据
            # 获取当前数据集中所有行人的 ID
            Pedlist = np.unique(data[1, :]).tolist()
            # Add the list of frameIDs to the frameList_data
            scene_id = directory.split('/')[-1]
            scene_list.append(scene_id)
            # 记录了当前数据集的每个帧包含了那些行人
            frameped_dict.append({})
            # 记录了每个行人的轨迹数据 （数据集，行人id，该行人的帧，对应帧下的xy数据）
            pedtrajec_dict.append({})
            pedlabel_dict.append({})
            for ind, pedi in enumerate(Pedlist):
                if ind % 100 == 0:
                    print(ind, len(Pedlist))
                # Extract trajectories of one person 抽取单人的轨迹数据
                FrameContainPed = data[:, data[1, :] == pedi]
                # Extract ped label
                Label = "Pedestrian"
                # Extract peds list
                FrameList = FrameContainPed[0, :].tolist()
                if len(FrameList) < 2:
                    continue
                # Initialize the row of the numpy array
                # -----------计算获取每个行人的轨迹数据
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
                # 保存对应的行人label维度
                pedlabel_dict[seti][pedi] = Label
        # ---------保存数据
        f = open(data_file, "wb")
        # 这两个对象序列化到文件中
        pickle.dump((frameped_dict, pedtrajec_dict, scene_list, pedlabel_dict), f, protocol=2)
        f.close()

    def data_preprocess_for_MLDGtask(self,setname):
        """
            基于data_preprocess_transformer将数据处理成MLDG可以运用的task结构类型
            完成原meta——task工作
        """
        # 第一步 加载对应数据集以及相应参数
        # 第二步 按场景分解获取对应batch数据
        self.data_preprocess_for_originbatch_split()
        trainbatch_meta = self.trainbatch_meta
        trainbatchnums_meta = self.trainbatchnums_meta
        cachefile = self.train_MLDG_batch_cache
        # 第三步 形成task-list
        task_list = []
        for seti, seti_batch_num in enumerate(trainbatchnums_meta):
            # 此处会有不同 ETH-UCY数据集直接移除相应的i即可 但是针对于SDD数据集,需要先聚合相同场景的代码
            if trainbatchnums_meta == 0 or trainbatchnums_meta[seti] == []:
                continue
            query_seti_id = list(range(len(trainbatchnums_meta)))
            #====ETH_UCY
            query_seti_id.remove(seti) # ETH5
            #====ETH_UCY
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
        # todo 最开始是按顺序获取task，获取完毕后打乱task  针对于sequential 不太适合 因为batch中的4个query task不一样，无法充分利用query数据
        # random.shuffle(task_list)
        batch_task_list = [task_list[i:i + self.args.query_sample_num] for i in range(0, len(task_list), self.args.query_sample_num)]
        batch_task_num = len(batch_task_list)
        self.pick_cache(trainbatch=batch_task_list,trainbatch_nums=batch_task_num,cachefile=cachefile)

    # =======实际具体任务

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, pedlabel_dict,data_index,scene_list,setname):
        """
        完成get_seq_from_index_balance / get_seq_from_index_balance_meta工作
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
                    batch_data_mass：多个（batch_data, Batch_id）
                    batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
                    nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                    seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                    nei_list_b：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
                    不同的只在于邻接矩阵的不同 单独一个 或则是 【3,20，257,257】
                    nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                    batch_pednum：list 表示该batch下每个时间窗口中的行人数量
        """
        batch_data_mass, batch_data, Batch_id = [], [], []
        ped_cnt, last_frame = 0, 0
        # 注意此处的skip 在不同数据集的差异
        if(setname=='train'): skip = self.trainskip
        else:skip=self.testskip
        # 全局处理 混合所有train的帧 形成的windows
        for i in range(data_index.shape[1]):
            '''
            仍然是以对应窗口序列划分 例如test有1443帧，则相应的可以划分处1443个时间窗口，但需要后期依据
            '''
            cur_frame, cur_set, _ = data_index[:, i]
            cur_scene = scene_list[cur_set]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            # 计算并获取对应起始帧（子轨迹）的结束帧，由于当前的子轨迹的结束帧可能会超过数据集的范围，因此使用try-expect语句块处理这种情况
            try:
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + (self.args.seq_length - 1) * skip[cur_set]]) #todo 尽量后续统一skip形式

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
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                               cur_frame, self.args.seq_length,
                                                                               skip[cur_set])
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
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
            print(self.args.dataset + '_' + setname + '_' + str(cur_pednum))
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
                    if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
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
        return batch_data_mass

    def massup_batch_HIN(self):
        """
            完成原massup_batch_HIN
        """
        pass

    def get_social_inputs_numpy_HIN(self):
        """
            完成原get_social_inputs_numpy_HIN
        """
        pass

