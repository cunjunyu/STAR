import os
import numpy as np
import pickle
import random
import json
from collections import Counter
from DataProcessor.DataProcessorFactory import DatasetProcessor_BASE

"""
    数据集分析
        1.实验设置:使用前10个间隔为30min的轨迹点来预测未来的5个轨迹点 seq_length,obs_length,pred_length
        2.测试设置：平均【RMSE,ADE】；最终单独点【RMSE，FDE】=> 皆需要从第1个点到第5个点的数据，现在先单独考量第5个数据
        3.归一化设置：此处的输入数据全部是归一化后的 需要注意 ！！ 可以反归一化，或则说在计算loss的时候进行反归一化
        4.数据格式：
        数据格式 [时间戳，lat，lon, 角度，速度， 航行距离，船类型，月，日，小时，开始轨迹点的lat，开始轨迹点的lon，mmsi]
        每次的时间戳(13位 以毫秒为单位的 Unix 时间戳) 如1628447286000-》1628447886000 每次间隔 600000=》10分钟 
        training: 27313 trajectory number; 6927 vessel number;49 type number =>list形式 684317个完整的15轨迹 
        ==》 需要确保此处截出来的数据量与原始的数据量级是一致的；原始的train为list[list[13]];
        第一个list表示有多少个轨迹数据，第二个list表示每条轨迹内的数据点，第三个表示每个数据点存储的数据信息
        原始的处理方法，将单条轨迹进行滑动选取 逐个选取完整的15点数据 
"""
def normalization(data=None, min_=None, max_=None):
    """
    对传入参数列表进行归一化操作==》
    :param max_:
    :param min_:
    :param data:
    :return:
    """
    data = float(data)
    new_a = (data - min_) / (max_ - min_)
    return new_a

def data_denormalize(norm_data):
    """
    Denormalizes a tensor using the provided min and max values.

    Args:
    - tensor (torch.Tensor): The tensor to be denormalized.
    - min_val (list or torch.Tensor): Min values for each feature (x and y).
    - max_val (list or torch.Tensor): Max values for each feature (x and y).

    Returns:
    - torch.Tensor: The denormalized tensor.
    """
    # predict / label (128,5,2) tensor
    # Ensure min_val and max_val are tensors
    min_val = [20.90883000, -133.29703000]
    max_val = [49.22927000, -60.68892000]
    data_denorm = [0,0]
    for i in range(2):
        data_denorm[i] = (norm_data[i] * (max_val[i] - min_val[i])) + min_val[i]
    return data_denorm


class DatasetProcessor_ship(DatasetProcessor_BASE):
    def __init__(self, args):
        """
        """

        args.sample_num = 1
        if args.denormalize == 'True':
            args.data_dir = args.save_base_dir + str(args.dataset) + '/' + str(args.test_set) + '/' + str(
                args.train_model) + '_' + str(args.stage) + str(args.denormalize) + '/'
            args.model_dir = args.save_base_dir + str(args.dataset) + '/' + str(args.test_set) + '/' + str(
                args.train_model) + '_' + str(args.stage) + str(args.denormalize) + '/' + str(args.param_diff) + '/'
            args.neighbor_thred = 0.03
            print("模型的新保存地址在"+args.model_dir)
        else:
            args.neighbor_thred = 0.001
        super().__init__(args=args)
        assert self.args.dataset == "Ship"
        print("正确完成真实数据集" + self.args.dataset + "的初始化过程")
        print("经纬度的邻居判断阈值为"+str(self.args.neighbor_thred))
        # 分析阈值参数
        """
        国际海事组织（IMO）的COLREGs（国际海上避碰规则）通常建议至少保持 1 海里的距离
        地球每度经线的距离约为 111 公里（在赤道处），而每度纬线的距离则因纬度而异，
        但在中纬度也大约是 111 公里。
        因此，1 海里（大约等于 1.852 公里）的距离大致对应于 1/60 度或 0.0167 度的经纬度变化。
        ===========
        Latitude: 0.0167/(49.22927-20.90883) = 0.00059（归一化后的值）
        Longitude: 0.0167/(|-60.68892|-|-133.29703|) =  -0.00023（归一化后的值）
        故而选取的阈值最少应该为 0.00059 即可以稍微取大一点 如=》 0.001
        """
        # 分析其他参数结构


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
    !重写 def data_preprocess_for_originbatch_split(self):
    =====batch数据形成类
    # def massup_batch(self,batch_data):
    # def get_social_inputs_numpy_HIN(self)
    """

    def data_preprocess_for_origintrajectory(self,args):
        """
            完成从最原始数据（csv）到初步处理的过程
            完成原traject——preprocess 工作
            ===============================
            1.获取数据存取位置
            2.配置基本实验设置
            3.划分训练和测试集数据

         """
        self.train_dir = r'./data/ship/train.json'
        self.test_dir = r'./data/ship/test.json'
        self.val_dir = r'./data/ship/val.json'
        self.max_min = r'./data/ship/max_min.json'
        """
        MAX_MIN: [ lat,lon,cog,sog,dis;
        MAX:[49.22927, -60.68892, 3.141592653589793, 30.695, 1327.35814], 
        MIN:[20.90883, -133.29703, 0.0, 0.0, 0.0]]
        test时需要用到？     
        train(args, Dtr, Val, LSTM_PATH)    
        test(args, Dte, LSTM_PATH, m, n) 
        """

        self.args.seq_length = 15
        self.args.obs_length = 10
        self.args.pred_length = 5
        self.Ship_skip = 600000   # 每次间隔 600000=》10分钟

        print("完成对ship数据的 训练测试划分")

    def data_preprocess_for_transformer(self, setname):
        """
            将数据处理成后续模型transformer的输入形式
            完成原dataPreprocess  工作 适合于无非区分域或则不需要区分域的形式
            dataPreprocess_sne
            获得 (frameped_dict, pedtrajec_dict, scene_list, pedlabel_dict)
        #=============================================
        完成四类型数据计算与存储：
        智能体的轨迹 pedtrajec_dict 形式pedtrajec_dict[pedi] = np.array(Trajectories);
        [int(frame),current_lat,current_lon,current_angle,current_speed]
        每帧包含的智能体标号 frameped_dict;  [vessel_id]
        智能体的类别 pedlabel_dict ; [mmsi,vessel_type]
        智能体的场景 scene_list;  ['Marine']
        为了后续代码的复用以及保持统一的形式 此处虽然没有多个场景 但仍然选择相同的单场景进行处理 ['Marine']
        """
        if setname == 'train':
            data_dir = self.train_dir
            data_file = self.train_data_file
        elif setname == 'test':
            data_dir = self.test_dir
            data_file = self.test_data_file
        print("处理__"+setname+"__数据")
        frameped_dict = []  # peds id contained in a certain frame 运用mmsi
        pedtrajec_dict = []  # trajectories of a certain ped 运用 time lat lon 角度 速度
        scene_list = ['Marine']  # 场景名称列表 no single
        pedlabel_dict = []  # vessel_type =》 与mmsi联系起来的 dict？ 存为tuple？
        # 角度 速度 航行距离
        with open(data_dir,'r') as f_read:
            data = json.load(f_read)
        # todo 反归一化 lat和lon的数据
            # Iterate over all trajectories in the train data
        if self.args.denormalize == 'True':
            print("反归一化数据开始")
            for trajectory in data:
                # Iterate over all time points in the trajectory
                for time_point in trajectory:
                    # Update the max and min values for each data point
                    time_point[1:3] = data_denormalize(time_point[1:3])
            print("反归一化数据完成")
        else:
            print("使用的是归一化后的数据")
        # 记录了每个行人的轨迹数据 （数据集，行人id，该行人的帧，对应帧下的xy数据）
        frameped_dict.append({})
        pedtrajec_dict.append({})
        pedlabel_dict.append({})
        # [时间戳，lat，lon, 角度，速度， 航行距离，船类型，月，日，小时，开始轨迹点的lat，开始轨迹点的lon，mmsi]
        for vessel_id,single_trajectory in enumerate(data):
            # 遍历 data 中的每一个元素（这里每个元素表示一个完整的轨迹）。
            single_trajectory_array = np.array(single_trajectory) # (longth,13)
            FrameList = single_trajectory_array[:,0].tolist()
            print("Array shape:", single_trajectory_array.shape)
            Trajectories = []
            mmsi = single_trajectory_array[0,-1]
            vessel_type = single_trajectory_array[0,6]
            for fi,frame in enumerate(FrameList):
                current_lat =  single_trajectory_array[single_trajectory_array[:,0]==frame,1]
                current_lon =  single_trajectory_array[single_trajectory_array[:,0]==frame,2]
                current_angle = single_trajectory_array[single_trajectory_array[:,0]==frame,3]
                current_speed = single_trajectory_array[single_trajectory_array[:,0]==frame,4]
                # 需要注意的是保留 0.41831165
                # .item() 用于将只包含一个元素的 NumPy 数组转换为一个标量值。
                Trajectories.append([int(frame),current_lat.item(),current_lon.item(),current_angle.item(),current_speed.item()])
                if int(frame) not in frameped_dict:
                    frameped_dict[0][int(frame)] = []
                frameped_dict[0][int(frame)].append(vessel_id)
            pedtrajec_dict[0][vessel_id] = np.array(Trajectories)
            pedlabel_dict[0][vessel_id] = [mmsi,vessel_type]
        f_save = open(data_file,"wb")
        pickle.dump((frameped_dict,pedtrajec_dict,scene_list,pedlabel_dict),f_save,protocol=2)
        f_save.close()

    # =======实际具体任务
    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, pedlabel_dict, data_index, scene_list,
                                   setname):
        batch = self.get_seq_from_index_balance_5(frameped_dict=frameped_dict, pedtraject_dict=pedtraject_dict,
                                                       pedlabel_dict=pedlabel_dict, scene_list=scene_list,
                                                       data_index=data_index, setname=setname)
        if self.args.train_model == 'new_star_ship':
            return batch
        elif self.args.train_model == 'star' or self.args.train_model == 'new_star':
            print("针对于该两种模型只用xy数据的情况 对原始数据进行再处理 将nodes_batch_b：(seq_length, num_Peds，4) 变回(seq_length, num_Peds，2)")
            new_batch_mass = []
            for id,single_batch in enumerate(batch):
                batch_data,batch_id = single_batch
                batch_old5, seq_list, nei_list, nei_num, batch_pednum = batch_data
                batch_new2 = batch_old5[:,:,:2]
                new_batchdata = (batch_new2, seq_list, nei_list, nei_num, batch_pednum)
                new_batch_mass.append((new_batchdata,batch_id,))
            return new_batch_mass
        elif self.args.train_model == 'new_star_hin':
            raise ValueError("针对于船的还未进行该类型的数据处理分析")
        else:
            raise ValueError("没有该种模型")

    def get_seq_from_index_balance_5(self, frameped_dict, pedtraject_dict, pedlabel_dict, data_index, scene_list,
                                   setname):
        """
            完成原始数据到batch的转变 步骤
            更改的一点在于 将其batch数据 由3转变为5了 即
            nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标 =》 (seq_length, num_Peds，4)后续需要依据不同的数据模型进行转换
        """
        batch_data_mass, batch_data, Batch_id = [], [], []
        ped_cnt, last_frame = 0, 0
        skip = self.Ship_skip
        present_pedi_list = []
        for i in range(data_index.shape[1]):
            cur_frame,cur_set,_ = data_index[:,i] #cur_frame为实际的在set中的frame，
            cur_scene = scene_list[cur_set]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            try:
                frameend_pedi = set(frameped_dict[cur_set][cur_frame+(self.args.seq_length-1)*skip])
            except:
                continue
            present_pedi = framestart_pedi | frameend_pedi
            # 观看结果可以发现 大部分在一个时间段内同时存在的更多的为单个，偶尔会有2个的；
            # print("第"+str(i)+"帧中包含的vessel为"+str(present_pedi))
            present_pedi_list.append(len(present_pedi))
            # 如果起始帧与结束帧没有重复的行人id，则抛弃该子轨迹 应该至少有一个
            if(framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            """
            针对由起始帧和结束帧确定的窗口序列以及行人并集，遍历行人，找到该行人在起始帧与结束帧之间存在的片段；若正好全程存在，则iffull为true，
            若有空缺，则iffull为False；ifexistobs标识obs帧是否存在，并删去太短的片段（小于5）；而后去除帧号，只保留这些行人的xy坐标；添加到traject中
            而后将滤除后的行人轨迹数据保留并拼接；batch-pednum为相应的不断累计不同时间窗口轨迹数据的总值，
            """
            for ped in present_pedi:
                # 此处与其他的不一样在于 返回的cur-trajec为(seq_length, return_len)即（15，5）其他数据集为（15,3）
                cur_trajec,iffull,ifexistobs = self.find_trajectory_fragment(trajectory=pedtraject_dict[cur_set][ped],
                                                startframe=cur_frame,seq_length=self.args.seq_length,skip=skip,return_len=5)
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                # 此时cur-trajec为固定的（20,3）则[:,1:]保留lon,lat,angle,speed数据，略去时间数据即（20,4）-》reshape为（20,1,4）数据
                cur_trajec = (cur_trajec[:,1:].reshape(-1,1,4),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            # 按照第二个维度进行拼接，即将同一个windows中行人数据present_pedi拼接在一起
            traject_batch = np.concatenate(traject,1)
            # 基于后续叠加各个windows中的行人数据
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]
            # 该windows中的行人数量
            cur_pednum = traject_batch.shape[1]
            # print(self.args.dataset + '_' + setname + '_' + str(cur_pednum))
            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame,)
            """
            如果以当前数据集以及相应的预测帧起始的窗口中包含超过512个行人的轨迹，则将其进行拆分为两个batch，如果处于256和512之间，
            将其打包成为一个batch；如果小于256，则相应的累加其他时间窗口的轨迹数据，直到batch里的行人数大于256,将其打包为一个batch             
            """
            # self.args.batch_around_ped 可以取小一点
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
                if batch_pednum > self.args.batch_around_ped:
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
                    """
                       输入：多个windows的数据 （windows-num，20，windows-ped，4）
                       batch_data_mass：多个（batch_data, Batch_id）
                       
                       batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
                       nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                       seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                       nei_list_b：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
                       nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                       batch_pednum：list 表示该batch下每个时间窗口中的行人数量
                    """
                    batch_data = self.massup_batch(batch_data)
                    if batch_data[4]:  # 即batch_pednum为空 则不添加该数据
                        batch_data_mass.append((batch_data, Batch_id,))
                    elif not batch_data[4]:
                        print('舍弃该数值')
                    last_frame = i
                    batch_data = []
                    Batch_id = []
                else:
                    # 累加
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
        # 统计分析 各个数值出现的情况
        count = Counter(present_pedi_list)
        for item,frequency in count.items():
            print(f"{setname} Value {item} appears {frequency} times")
        if last_frame < data_index.shape[1] - 1 and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数据')
            # batch_data_mass.append((batch_data, Batch_id,)
        return batch_data_mass

    def massup_batch(self, batch_data):
        """
        核心是将来自不同时间窗口的行人轨迹数据聚合成一个大批次，同时保留每帧每个行人的邻居关系和其他相关信息
        input: list[traject_batch（20,ped-nums-in-windows,4）]
        output: (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
        """
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]
        # 子轨迹的位置序列 (seq_length, num_Peds) 掩码序列01
        # 创建一个这样的“空”数组是有用的，特别是当你计划稍后通过添加列来动态地填充数据时
        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 4))
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
            # 相应的将该时间窗口的数据 batch添加进nodes_batch_b 按第二维度1 即行人的维度 （20，num-ped，2）
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
        识别和量化行人在每一帧中基于空间距离的社交互动=>同一个windows下
        input: traject_batch（20,ped-nums-in-windows,4）
        output: seq_list, nei_list, nei_num
            num_Peds：输入数据 inputnodes 中的行人数量。
            seq_list：一个零矩阵，用于标识每一帧中每个行人是否存在（有数据）。
            nei_list 和 nei_num：分别初始化为零矩阵，用于存储每一帧中行人的邻居关系和每个行人的邻居数量。

        """
        num_Peds = inputnodes.shape[1]
        seq_list = np.zeros((inputnodes.shape[0], num_Peds))

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
                """
                    以两个行人邻居分析为例：选的是帧 
                        可以明确地是：需要选取的是pedi和pedj都有数据 又满足邻居距离阈值的帧 
                        select：最初是一个布尔数组，表示哪些帧中行人 pedi 和 pedj 都有数据。
                        select_dist：是一个布尔数组，它标记了那些行人 pedi 和 pedj 之间的相对坐标超出了邻居距离阈值的帧。
                """
                nei_num[select, pedi] -= select_dist
                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num

    # =====后续开发================

    def data_preprocess_for_MLDGtask(self,setname):
        # 后续开发
        pass

    def data_preprocess_for_originbatch_split(self):
        # 后续开发
        pass

    def massup_batch_HIN(self):
        # 后续开发
        pass

    def get_social_inputs_numpy_HIN(self):
        # 后续开发
        pass




