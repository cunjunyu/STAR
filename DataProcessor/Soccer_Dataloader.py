import pickle
import os
import numpy as np
from DataProcessor.NBA_Dataloader import NBA_Dataloader


class Soccer_Dataloader_HIN(NBA_Dataloader):
    def __init__(self, args):
        # super(Soccer_Dataloader, self).__init__(args) 不需要父类的初始化方法
        # dataset Soccer
        # 负责两类数据处理 第一类数据处理 相应的单独soccer数据实验结果，第二类 用nba数据当训练集，在soccer数据集上进行测试
        # 需要注意的是 由于单独的soccer数据太少了，此处直接运用拆分形式
        self.args = args
        """
        You'll see the data goes from 0 to 1 on each axis.
        The coordiante (0,0) is the top left, (1,1) is the bottom right, and (0.5,0.5) is the kick off point.
        The dimensions of the field are the same for both games: 105x68 meters.
        """
        self.args.seq_length = 15
        self.args.obs_length = 5
        self.args.pred_length = 10
        self.train_dir = ['./data/NBA/nba/source/CLE', './data/NBA/nba/source/GSW', './data/NBA/nba/source/NYK',
                          './data/NBA/nba/source/OKC', './data/NBA/nba/source/SAS']
        self.test_dir = '../data/soccer/all_trajs.npy'
        self.train_skip = [10, 10, 10, 10, 10]
        self.test_skip = [10]  # 0.04s一帧 故而降采样为0,4s

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
        self.test_batch, self.test_batchnums = self.load_split_data(self.test_batch_cache)
        print('Total number of test batches:', self.test_batchnums)

        if self.args.stage == 'origin':
            print("Preparing origin data batches.")
            if not (os.path.exists(self.train_batch_cache) or os.path.exists(self.train_batch_cache_split)):
                self.traject_dict = self.load_dict(self.train_data_file)
                self.dataPreprocess('train')
            self.train_batch, self.train_batchnums = self.load_split_data(self.train_batch_cache)
            print('Total number of training origin batches:', self.train_batchnums)

        elif self.args.stage == 'meta':
            if not (os.path.exists(self.train_MLDG_batch_cache) or os.path.exists(self.train_MLDG_batch_cache_split)):
                print("process train meta cpkl")
                self.traject_dict = self.load_dict(self.train_data_file)
                self.MLDG_TASK(setname='train')
            self.train_batch_task, self.train_batch_tasknums = self.load_split_data(self.train_MLDG_batch_cache)
            print('Total number of training MLDG task batches :', len(self.train_batch_task))

        elif self.args.stage == 'MVDG' or self.args.stage == 'MVDGMLDG':
            if not (os.path.exists(self.train_MVDG_batch_cache) or os.path.exists(self.train_MVDG_batch_cache_split)):
                self.traject_dict = self.load_dict(self.train_data_file)
                self.MVDG_TASK(setname='train')
            self.train_batch_MVDG_task, self.train_batch_MVDG_tasknums = self.load_split_data(
                self.train_MVDG_batch_cache)
            print('Total number of training MVDG task batches :', len(self.train_batch_MVDG_task))

    # 通用的NBA-dataloader方法 此处不在赘述 只简单列举
    """
    def load_dict(self, data_file):
    def load_cache(self, data_file):
    def pick_split_data(self,train_batch,cachefile,batch_size):
    def load_split_data(self,cachefile):
    def get_train_batch(self, idx):
    def get_test_batch(self, idx):
    def reset_batch_pointer(self, set, valid=False):
    ======================================
    def get_data_index(self, data_dict, setname, ifshuffle=True):
    def get_data_index_meta(self, seti, data_dict, setname, ifshuffle=True):
    def rotate_shift_batch(self, batch_data, ifrotate=True):
    def MLDG_TASK(self,setname):
    def MVDG_TASK(self,setname):
    """

    def traject_preprocess(self, setname):
        # 处理数据 分割train-test；实验内容应该包括：train-test同源结果，不同源结果
        trajec_dict = {}
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
            for seti, directory in enumerate(data_dirs):
                all_trajs_data_root = os.path.join(directory, "all_trajs.npy")
                self.all_trajs = np.load(all_trajs_data_root)
                self.all_trajs /= (94 / 28)  # Turn to meters
                self.all_trajs = self.all_trajs.transpose(0, 2, 1, 3)
                # all_trajs [803,10,15,2] [B,ped-num,time-seq,xy]
                trajec_dict[seti] = self.all_trajs
                # 'GSW': 803  'OKC': 705  ‘SAS’ ：895, 'CLE' : 717
        elif setname == 'test':
            data_dirs = self.test_dir
            data_file = self.test_data_file
            self.all_trajs = np.load(data_dirs)
            # 数据转换，从0-1转换为105*68的标准数据？
            # 被处理成完整的格式（1476,15,22,2）
            length, width = 105, 68  # 场地尺寸
            # 转换坐标
            self.all_trajs[..., 0] *= length  # 将 x 坐标转换回原始尺寸
            self.all_trajs[..., 1] *= width  # 将 y 坐标转换回原始尺寸
            self.all_trajs = self.all_trajs.transpose(0, 2, 1, 3)
            trajec_dict[0] = self.all_trajs
            # all_trajs [1476,15,22,2] [B,ped-num,time-seq,xy]
        f = open(data_file, "wb")
        pickle.dump(trajec_dict, f, protocol=2)
        f.close()

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname == 'train':
            # 不一样 没有对应的frame-list与ped-list
            traject_dict = self.traject_dict
            cachefile = self.train_batch_cache
            shuffle = True
            data_now = 'NBA'
        else:
            traject_dict = self.test_traject_dict
            cachefile = self.test_batch_cache
            shuffle = False
            data_now = 'soccer'
        data_index = self.get_data_index(traject_dict, setname, ifshuffle=shuffle)
        # traject——dict不一样
        # 因为需要同时处理 NBA和Soccer 故而 get_seq_from_index_balance 以及相关的函数应该都是通用的
        train_batch = self.get_seq_from_index_balance(traject_dict, data_index, setname)
        train_batchnums = len(train_batch)
        print(str(data_now) + 'batch_nums:' + str(train_batchnums))
        f = open(cachefile, "wb")
        pickle.dump((train_batch, train_batchnums), f, protocol=2)
        f.close()

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
                batch_data = self.massup_batch(batch_data, setname)
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
            batch_data = self.massup_batch(batch_data, setname)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数值')
            # batch_data_mass.append((batch_data, Batch_id,))
        return batch_data_mass

    def massup_batch(self, batch_data, setname):
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
        nei_list_b = np.zeros((relation_num, self.args.seq_length, num_Peds, num_Peds))
        nei_num_b = np.zeros((relation_num, self.args.seq_length, num_Peds))

        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            num_Ped = batch.shape[1]
            # todo 重写get-social-inputs-numpy
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch, relation_num, setname)
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, :, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, :, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

    def get_social_inputs_numpy(self, inputnodes, relation_num, setname):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]
        relation_matrix = np.zeros((num_Peds, num_Peds))
        # 单独考虑两队球员的轨迹 todo 在SDD中针对性的改为依据label
        if setname == 'train':
            ped_thred = 5
        elif setname == 'test':
            ped_thred = 11
        for i in range(num_Peds):
            for j in range(num_Peds):
                # 球队1
                if i < ped_thred and j < ped_thred:
                    relation_matrix[i][j] = 0
                # 球队2
                elif i >= ped_thred and j >= ped_thred:
                    relation_matrix[i][j] = 1
                # 球队1和球队2之间的关系
                elif i < ped_thred and j >= ped_thred:
                    relation_matrix[i][j] = 2
                elif i >= ped_thred and j < ped_thred:
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


class Soccer_Dataloader_NOHIN(Soccer_Dataloader_HIN):

    def __init__(self, args):
        # super(Soccer_Dataloader, self).__init__(args) 不需要父类的初始化方法
        # dataset Soccer
        # 负责两类数据处理 第一类数据处理 相应的单独soccer数据实验结果，第二类 用nba数据当训练集，在soccer数据集上进行测试
        # 需要注意的是 由于单独的soccer数据太少了，此处直接运用拆分形式
        self.args = args
        """
        You'll see the data goes from 0 to 1 on each axis.
        The coordiante (0,0) is the top left, (1,1) is the bottom right, and (0.5,0.5) is the kick off point.
        The dimensions of the field are the same for both games: 105x68 meters.
        """
        self.args.seq_length = 15
        self.args.obs_length = 5
        self.args.pred_length = 10
        self.train_dir = ['./data/NBA/nba/source/CLE', './data/NBA/nba/source/GSW', './data/NBA/nba/source/NYK',
                          './data/NBA/nba/source/OKC', './data/NBA/nba/source/SAS']
        self.test_dir = '../data/soccer/all_trajs.npy'
        self.train_skip = [10, 10, 10, 10, 10]
        self.test_skip = [10]  # 0.04s一帧 故而降采样为0,4s

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
        self.test_batch, self.test_batchnums = self.load_split_data(self.test_batch_cache)
        print('Total number of test batches:', self.test_batchnums)

        if self.args.stage == 'origin':
            print("Preparing origin data batches.")
            if not (os.path.exists(self.train_batch_cache) or os.path.exists(self.train_batch_cache_split)):
                self.traject_dict = self.load_dict(self.train_data_file)
                self.dataPreprocess('train')
            self.train_batch, self.train_batchnums = self.load_split_data(self.train_batch_cache)
            print('Total number of training origin batches:', self.train_batchnums)

        elif self.args.stage == 'meta':
            if not (os.path.exists(self.train_MLDG_batch_cache) or os.path.exists(self.train_MLDG_batch_cache_split)):
                print("process train meta cpkl")
                self.traject_dict = self.load_dict(self.train_data_file)
                self.MLDG_TASK(setname='train')
            self.train_batch_task, self.train_batch_tasknums = self.load_split_data(self.train_MLDG_batch_cache)
            print('Total number of training MLDG task batches :', len(self.train_batch_task))

        elif self.args.stage == 'MVDG' or self.args.stage == 'MVDGMLDG':
            if not (os.path.exists(self.train_MVDG_batch_cache) or os.path.exists(self.train_MVDG_batch_cache_split)):
                self.traject_dict = self.load_dict(self.train_data_file)
                self.MVDG_TASK(setname='train')
            self.train_batch_MVDG_task, self.train_batch_MVDG_tasknums = self.load_split_data(
                self.train_MVDG_batch_cache)
            print('Total number of training MVDG task batches :', len(self.train_batch_MVDG_task))

    """
    # 通用的NBA-dataloader方法 此处不在赘述 只简单列举
    def load_dict(self, data_file):
    def load_cache(self, data_file):
    def pick_split_data(self,train_batch,cachefile,batch_size):
    def load_split_data(self,cachefile):
    def get_train_batch(self, idx):
    def get_test_batch(self, idx):
    def reset_batch_pointer(self, set, valid=False):
    ======================================
    def get_data_index(self, data_dict, setname, ifshuffle=True): 已经由NBA重新写过了 符合数据要求
    def get_data_index_meta(self, seti, data_dict, setname, ifshuffle=True):
    def rotate_shift_batch(self, batch_data, ifrotate=True):
    def MLDG_TASK(self,setname):
    def MVDG_TASK(self,setname):
    ======================================
    # 通用的NBA-dataloader_HIN方法 此处不在赘述 只简单列举
    # 只是分别预处理 两者数据 不影响  # 未修改 与soccer-dataloader-hin一样
    def traject_preprocess(self,setname):
    def dataPreprocess(self, setname):
    def get_seq_from_index_balance(self, traject_dict, data_index, setname):
    """

    # 主要区别在于去除相应的HIN关系 只保留最原始的mass
    # 修改了 改为eth-ucy的原始形式
    def massup_batch(self, batch_data, setname):
        '''
        用回最原始的eth-ucy中的mass-up
        Massed up data fragements in different time window together to a batch
        '''
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]

        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 2))
        # 与SDD统一格式 都为 relation seq-length num-peds num-peds
        nei_list_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        nei_num_b = np.zeros((self.args.seq_length, num_Peds))

        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            num_Ped = batch.shape[1]
            # todo 重写get-social-inputs-numpy
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch, setname)
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

    # 修改了 改为eth-ucy的原始形式
    def get_social_inputs_numpy(self, inputnodes, setname):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]
        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
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
                        abs(relative_cord[:, 1]) > self.args.neighbor_thred)
                # 自己与自己需要删除

                nei_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num
