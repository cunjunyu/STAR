import os
import numpy as np
import pickle
import random

from DataProcessor.DataProcessorFactory import DatasetProcessor_BASE
from DataProcessor.NBA_preprocess import DatasetProcessor_NBA

class DatasetProcessor_Soccer(DatasetProcessor_NBA):
    def __init__(self, args):
        super().__init__(args=args)
        assert  self.args.dataset == "Soccer"
        print("正确完成真实数据集"+self.args.dataset+"的初始化过程")
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

        You'll see the data goes from 0 to 1 on each axis.
        The coordiante (0,0) is the top left, (1,1) is the bottom right, and (0.5,0.5) is the kick off point.
        The dimensions of the field are the same for both games: 105x68 meters.
        """
        self.args.seq_length = 15
        self.args.obs_length = 5
        self.args.pred_length = 10
        self.train_dir = ['./data/NBA/nba/source/CLE', './data/NBA/nba/source/GSW', './data/NBA/nba/source/NYK',
                          './data/NBA/nba/source/OKC', './data/NBA/nba/source/SAS']
        self.test_dir = './data/soccer/all_trajs.npy'
        self.train_skip = [10, 10, 10, 10, 10]
        self.test_skip = [10]  # 0.04s一帧 故而降采样为0,4s

    def data_preprocess_for_transformer(self, setname):
        """
        Soccer 与NBA类似 但也有不同

        """
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
        print(str(setname) +'Soccer batch_nums:' + str(trainbatchnums))
        self.pick_cache(trainbatch=trainbatch, trainbatch_nums=trainbatchnums, cachefile=cachefile)

    # =======实际具体任务
    def get_social_inputs_numpy_HIN(self, inputnodes, relation_num, setname):
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

    # 同样的 get_seq_from_index_balance_origin需要后续验证
    def get_seq_from_index_balance_origin(self, traject_dict, data_index, setname):
        """
        留待后续开发
        """
        # NBA的需要进行验证
        # skip 等其他还是不一样的 需要思考开发一下
        # 虽然相应的massup——batch和get_social_inputs_numpy应该是一样的
        pass




