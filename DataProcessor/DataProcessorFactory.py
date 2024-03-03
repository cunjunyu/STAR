import os
import numpy as np
import pickle
import random
from abc import ABC, abstractmethod


# from DataProcessor.ETH_UCY_preprocess import DatasetProcessor_ETH_UCY


class DatasetProcessor_BASE(ABC):
    def __init__(self, args):
        """
        有没有一种可能 即将这个过程放到BASE中 但后期调用的函数是新的函数 这样不用为每个进行重复创建__init__
        在对象创建过程中完成的
        全局对象 self.args
        各通用文件名称 之前的文件命名存在保存过多份数据的问题
        1. 最基础的喂给transformer的数据可以保存 在通用的外围数据 即存于 ./output/eth_ucy/test_set/model中即可==》
        进一步分析HIN和非HIN的数据仍然不一样 故而仍然不同 最好还是存在结合model层
        2. 考虑到代码数据量过大的问题 提前实现相同的分散保存体系
        ==========================================
            依据输入参数完整整体的数据选择调用工作
            完成原Init的工作
            先确定模型和类型 再继续生成对应的数据保存位置等
        =============================================
        1. 完成数据最初处理  读取数据  划分训练  测试
        2. 完成数据从origin 到 等待batch处理的形式（frameped_dict, pedtrajec_dict, scene_list, pedlabel_dict）
        3. 完成从batch到origin，mldg，mvdg处理的形式
        """
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
        self.data_preprocess_for_origintrajectory(args=self.args)
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
            self.test_frameped_dict, self.test_pedtraject_dict, self.test_scene_list, \
            self.test_pedlabel_dict = self.load_dict(self.test_data_file)
            self.data_preprocess_for_originbatch('test')
        self.test_batch, self.test_batchnums = self.load_cache(self.test_batch_cache)
        print('test的数据批次总共有', self.test_batchnums)

        # ---依据不同的情况选择对应的train数据集batch
        self.train_frameped_dict, self.train_pedtraject_dict, self.train_scene_list, \
        self.train_pedlabel_dict = self.load_dict(self.train_data_file)
        if self.args.stage == 'origin' and self.args.phase == 'train':
            print("origin::处理对应的origin-train-batch数据")
            # ----origin_for_transformer----------------
            self.train_origin_batch_cache = os.path.join(self.args.data_dir, "train_origin_batch_cache.cpkl")
            self.train_origin_batch_cache_split = os.path.join(self.args.data_dir, "train_origin_batch_cache_0.cpkl")
            if not (os.path.exists(self.train_origin_batch_cache) or os.path.exists(
                    self.train_origin_batch_cache_split)):
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

    # 数据读取存储类 明确读取的不同类型
    def load_dict(self, data_file):
        """
         数据读取存储类 明确读取的不同类型 尽量同时可以保存对应的数据量
        """
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        traject_dict = raw_data

        return traject_dict

    def pick_cache(self, trainbatch, trainbatch_nums, cachefile):
        """
        依据不同的数据集以及是否HIN等关系 选择对应的打包方法 是单独打包成一个还是打包成多个
        此处的trainbatch——nums为阈值 batch—size为代码
        """
        # 需要有一段依据不同数据集 不同阶段 HIN与否来判断是否需要进行打包
        if self.args.dataset == 'SDD' and self.args.HIN:
            threshold_for_pick = 40;
            batch_size = 40;
        elif self.args.dataset == 'NBA':
            threshold_for_pick = 50;
            batch_size = 50;
        elif self.args.dataset == 'Ship':
            # 文件数量较少的情况下 会有较好的结果
            threshold_for_pick = 100;
            batch_size = 100;
        else:
            threshold_for_pick = 200;
            batch_size = 200;
        if trainbatch_nums < threshold_for_pick:
            f = open(cachefile, "wb")
            pickle.dump(trainbatch, f, protocol=2)
            f.close()
        else:
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
                print('pickle data to ' + str(cachefile))
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

    def load_cache(self, cachefile):
        """
        注意当数据过大时如何处理 SDD中相应的有 pick_split_data load_split_data
        """
        """
                注意当数据过大时如何处理 SDD中相应的有 pick_split_data load_split_data
                """
        # batch_size = 1000  # 每个批次的大小
        # num_batches = 10  # 分批的总数
        # 第一步 判断对应的存储文件是简单的cachefile还是复杂的cachefile——0
        filename_without_extension, file_extension = os.path.splitext(cachefile)
        cachefile_split = f"{filename_without_extension}_{0}.cpkl"
        if os.path.exists(cachefile):
            f = open(cachefile, 'rb')
            raw_data = pickle.load(f)
            f.close()
            return raw_data, len(raw_data)
        elif os.path.exists(cachefile_split):
            # 第二步 统计每个目录下有多少个分开的文件夹
            # 去除最后一个部分，只保留前面的部分 即上级目录
            directory_path = os.path.dirname(cachefile_split)
            # 选取相应的文件夹中的前缀名 eg:test_batch_cache
            file_prefix = filename_without_extension.split('/')[-1]
            # directory_path,file_prefix = os.path.split(cachefile_split)
            file_list = [filename for filename in os.listdir(directory_path) if
                         filename.startswith(file_prefix) and filename.endswith(file_extension)]
            num_batches = len(file_list)
            combined_trainbatch = []
            # 第三步 分文件按顺序读出文件 并拼接
            for i in range(num_batches):
                cachefile = f"{filename_without_extension}_{i}.cpkl"
                print('load data from :' + str(cachefile))
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

    # batch获取类数据
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
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0

        pass

    def get_data_index_single(self, seti, data_dict, setname, ifshuffle=True):
        """
        输入的data-dict是个list，只有单个场景的数据，时间序列，其存储了每个场景从第一帧到最后一帧（固定间隔）的行人标号
        setname：train/test
        完成get_data_index / get_data_index_meta工作
        Get the dataset sampling index.
        data-dict：集合，包含了多个场景，每个场景是一个时间序列，其存储了每个场景从第一帧到最后一帧（固定间隔）的行人标号
        setname：train/test
        其主要作用是返回：所有帧 ID 和它们所属的数据集 ID、数字化后的帧 ID 存储在一个 3 x N 的 NumPy 数组 data_index 中
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

    def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip, return_len=3):
        '''
        Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
        核心是在一个给定的轨迹数据集中根据起始帧号和序列长度提取对应的轨迹片段，
        如果数据不完整，则在缺失的部分填充零。这种处理方式在轨迹分析和运动预测等领域非常常见。
        '''
        return_trajec = np.zeros((seq_length, return_len))
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
        return_trajec[offset_start:offset_end + 1, :return_len] = candidate_seq
        # 返回的轨迹长度在观测帧处存在对应的值
        if return_trajec[self.args.obs_length - 1, 1] != 0:
            ifexsitobs = True
        # 返回的轨迹长度大于等于对应要求的序列长度，
        if offset_end - offset_start >= seq_length - 1:
            iffull = True

        return return_trajec, iffull, ifexsitobs

    def massup_batch(self, batch_data):
        """
            完成原massup_batch
        """
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
        """
            完成原get_social_inputs_numpy

        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        对于每对行人 (i, j)，计算它们之间的相对坐标，如果相对坐标中任意一个分量的绝对值超过了阈值 self.args.neighbor_thred，则认为它们之间没有邻居关系
        """
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
                """
                select 最初是一个布尔数组，表示哪些帧中行人 pedi 和 pedj 都有数据。
                通过将 select 中为 True 的元素设置为 select_dist 的相应值，
                select 现在表示那些行人 pedi 和 pedj 既有数据又满足邻居距离阈值的帧。
                """
                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
                # 主要母的就是填满对应的0-1矩阵
        return seq_list, nei_list, nei_num

    # 无需重复模块化设计 ================================
    def data_preprocess_for_originbatch(self, setname):
        """
            将四种类型的数据 转换成 batch处理的形式
        """
        print(setname)
        if setname == 'train':
            frameped_dict = self.train_frameped_dict
            pedtraject_dict = self.train_pedtraject_dict
            pedlabel_dict = self.train_pedlabel_dict
            scene_list = self.train_scene_list
            cachefile = self.train_origin_batch_cache
            shuffle = True
        else:
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            pedlabel_dict = self.test_pedlabel_dict
            scene_list = self.test_scene_list
            cachefile = self.test_batch_cache
            shuffle = False

        data_index = self.get_data_index(data_dict=frameped_dict, setname=setname, ifshuffle=shuffle)
        trainbatch = self.get_seq_from_index_balance(frameped_dict=frameped_dict, pedlabel_dict=pedlabel_dict,
                                                     pedtraject_dict=pedtraject_dict, scene_list=scene_list,
                                                     data_index=data_index, setname=setname)
        trainbatchnums = len(trainbatch)
        print(str(setname) + 'origin_batch_num:' + str(trainbatchnums))
        self.pick_cache(trainbatch=trainbatch, trainbatch_nums=trainbatchnums, cachefile=cachefile)

    # 针对于分领域的获取batch
    def data_preprocess_for_originbatch_split(self):
        # 第一步 加载对应数据集以及相应参数
        frameped_dict = self.train_frameped_dict
        pedtraject_dict = self.train_pedtraject_dict
        pedlabel_dict = self.train_pedlabel_dict
        scene_list = self.train_scene_list
        # cachefile = self.train_MVDG_batch_cache
        shuffle = True
        trainbatch_meta = []
        trainbatchnums_meta = []
        # 第二步 按场景分解获取对应batch数据
        for seti, seti_frameped_dict in enumerate(frameped_dict):
            trainbatch_meta.append({})
            # 只需要在此处将data-index按相应的场景分开即可 此处data-index传入了对应的seti 故而出来的data-index会与结果有较好的对应
            train_index = self.get_data_index_single(seti=seti, data_dict=seti_frameped_dict, setname='train',
                                                     ifshuffle=shuffle)
            train_batch = self.get_seq_from_index_balance(frameped_dict=frameped_dict, pedtraject_dict=pedtraject_dict,
                                                          pedlabel_dict=pedlabel_dict, data_index=train_index,
                                                          scene_list=scene_list, setname='train')
            trainbatchnums = len(train_batch)
            # list（场景号） - list（windows号） -tuple （）
            trainbatch_meta[seti] = train_batch
            trainbatchnums_meta.append(trainbatchnums)
        self.trainbatch_meta = trainbatch_meta
        self.trainbatchnums_meta = trainbatchnums_meta

    # 借助于MLDG-task进行简化 模块化设计 基于MLDG的后续MVDG步骤是统一的
    def data_preprocess_for_MVDGtask(self, setname):
        """
        基于data_preprocess_transformer将数据处理成MVDG可以运用的task结构类型
        完成原MVDG-task工作
        """
        # 第一步 加载对应数据集以及相应参数
        # 第二步 按场景分解获取对应batch数据
        # 第三步 形成task-list
        # 前面3步由MLDG的task完成 故而只需要确认MLDG相应的文件是否存在 如果存在直接解压 不存在则计算一遍而后继续后续
        self.train_MLDG_batch_cache = os.path.join(self.args.data_dir, "train_MLDG_batch_cache.cpkl")
        self.train_MLDG_batch_cache_split = os.path.join(self.args.data_dir, "train_MLDG_batch_cache_0.cpkl")
        if not (os.path.exists(self.train_MLDG_batch_cache) or os.path.exists(self.train_MLDG_batch_cache_split)):
            # 不需要load——dict 因为相应的load-dict工作在进行MVDG判断的时候已经存在
            self.data_preprocess_for_MLDGtask('train')
        self.train_batch_MLDG_task, self.train_batch_MLDG_tasknums = self.load_cache(self.train_MLDG_batch_cache)
        # 第四步：形成mvdg-list
        cachefile = self.train_MVDG_batch_cache
        batch_task_list = self.train_batch_MLDG_task
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
        self.pick_cache(trainbatch=new_batch_task_list, trainbatch_nums=new_batch_task_list_num, cachefile=cachefile)

    # 下面的方法都必须重新实现------------------------------------------
    @abstractmethod
    def data_preprocess_for_origintrajectory(self):
        """
            完成从最原始数据（csv）到初步处理的过程
            完成原traject——preprocess 工作
            该数据 整体只存一份
         """
        pass

    @abstractmethod
    def data_preprocess_for_transformer(self):
        """
            将数据处理成后续模型transformer的输入形式
            完成原dataPreprocess  工作 适合于无非区分域或则不需要区分域的形式
            dataPreprocess_sne
        """

        pass

    # 不同的数据集域划分的方式不一样 故而MLDG的生成也略微不一样 故而需要重写
    @abstractmethod
    def data_preprocess_for_MLDGtask(self):
        """
            基于data_preprocess_transformer将数据处理成MLDG可以运用的task结构类型
            完成原meta——task工作
        """
        pass

    @abstractmethod
    def get_seq_from_index_balance(self):
        """
            完成get_seq_from_index_balance / get_seq_from_index_balance_meta工作
        """
        pass

    # 为异质图而设计
    @abstractmethod
    def massup_batch_HIN(self):
        """
            完成原massup_batch_HIN
        """
        pass

    @abstractmethod
    def get_social_inputs_numpy_HIN(self):
        """
            完成原get_social_inputs_numpy_HIN
        """
        pass


"""
class DataProcessorFactory:
    @staticmethod
    def get_processor(dataset_type):
        if dataset_type == "ETH_UCY":
            return DatasetProcessor_ETH_UCY()
        elif dataset_type == "SDD":
            return DatasetProcessor_SDD()
        elif dataset_type == "NBA":
            return DatasetProcessor_NBA()
        elif dataset_type == "Soccer":
            return DatasetProcessor_Soccer()
        elif dataset_type == "ship":
            return DatasetProcessor_ship()
        else:
            raise ValueError("Unknown dataset type")
"""
