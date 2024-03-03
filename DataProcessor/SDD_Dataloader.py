import os
import pickle
import random
import time
import numpy as np
# import torch
import pandas as pd
import os
from src.Visual import SDD_traj_vis
# import cv2
# from copy import deepcopy


#  为SDD数据处理准备 以及SDD数据的统计信息处理 为introduction做准备

def load_SDD(path='./data/SDD/', mode='train'):
    '''
    Loads data from Stanford Drone Dataset. Makes the following preprocessing:
    -filter out unnecessary columns (e.g. generated, label, occluded)
    -filter out non-pedestrian
    -filter out tracks which are lost
    -calculate middle point of bounding box
    -makes new unique, scene-dependent ID (column 'metaId') since original dataset resets id for each scene
    -add scene name to column for visualization
    -output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

    before data needs to be in the following folder structure
    data/SDD/mode               mode can be 'train','val','test'
    |-bookstore_0
        |-annotations.txt
        |-reference.jpg
    |-scene_name
        |-...
    :param path: path to folder, default is 'data/SDD'
    :param mode: dataset split - options['train', 'test', 'val']
    :return: DataFrame containing all trajectories from dataset split
    '''
    assert mode in ['train', 'val', 'test']

    path = os.path.join(path, mode)
    # 获取所有场景的列表 此处相应的是字符还是数字？？
    scenes = os.listdir(path)
    SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    data = []
    print('loading ' + mode + ' data')
    for scene in scenes:
        scene_path = os.path.join(path, scene, 'annotations.txt')
        scene_df = pd.read_csv(scene_path, header=0, names=SDD_cols, delimiter=' ')
        # Calculate center point of bounding box
        scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
        scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
        # drop non-pedestrians 舍弃该行 所有类型的agent都保留下来
        # scene_df = scene_df[scene_df['label'] == 'Pedestrian']
        scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
        # 使用drop方法，将数据框中名为xmin、xmax、ymin、ymax、occluded、generated，lost的列删除
        scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost'])
        scene_df['sceneId'] = scene
        # new unique id “rec&trackId” by combining scene_id and track_id 舍弃不需要 应该
        # 使用了列表推导式，遍历了scene_df的sceneId和trackId两列，将它们拼接起来，并用下划线连接，形成一个新的字符串，最后将所有字符串组成一个新的列表，作为rec&trackId列的值。
        # 代码中使用了zfill()方法对trackId进行了填充，使得字符串的长度为4，这样可以保证rec&trackId列中的所有字符串长度都相同
        scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
                                   zip(scene_df.sceneId, scene_df.trackId)]
        # 最终的数据格式 【trackID，x,y,frame,label,sceneID]
        data.append(scene_df)
    # 使用concat将一个列表中的多个数据框合并到一起，并重新生成索引
    data = pd.concat(data, ignore_index=True)
    # 创建rec-trackID2metaId的字典，用于将每个唯一的rec&trackId映射到一个唯一的metaId（整数编号）
    rec_trackId2metaId = {}
    for i, j in enumerate(data['rec&trackId'].unique()):
        rec_trackId2metaId[j] = i
    data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
    data = data.drop(columns=['rec&trackId'])
    # 相应的一个多个场景的数据列表
    return data


def mask_step(x, step):
    """
    Create a mask to only contain the step-th element starting from the first element. Used to downsample
    mask_step函数用于创建一个布尔类型的掩码（mask），这个掩码用于选择数据中每隔step个元素中的一个
    """
    mask = np.zeros_like(x)
    mask[::step] = 1
    return mask.astype(bool)


def downsample_all_metaId(df, step):
    """
    Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
    df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
    pedestrian (metaId)
    函数根据metaId列对数据框进行分组，然后对每个分组应用mask_step函数，得到一个布尔类型的掩码，用于选择每个分组中每隔step个元素中的一个。
    最后，函数将所有掩码合并起来，得到一个整体的掩码，并使用这个掩码对原始数据框进行选择，得到降采样后的数据框，并将其返回。
    :param df: pandas DataFrame - necessary to have column 'metaId'
    :param step: int - step size, similar to slicing-step param as in array[start:end:step]
    :return: pd.df - downsampled
    """
    mask = df.groupby(['trackId'])['trackId'].transform(mask_step, step=step)
    return df[mask]


def downsample_all_frame(df, step):
    """
    函数首先统计帧的数目，而后从0开始，隔12取一次，直到末尾；
    而后基于该筛选出的帧，取出对应数据 并重新赋值metaID
    """
    columns_counts = df.nunique()
    frame_max = columns_counts['frame']
    max_divisible = (frame_max // step) * step
    numbers_every_12 = [i for i in range(0, max_divisible + 1, step)]
    filtered_data = df[df['frame'].isin(numbers_every_12)]
    rec_trackId2metaId = {}
    for i, j in enumerate(filtered_data['metaId'].unique()):
        rec_trackId2metaId[j] = i
    filtered_data.loc[:, 'metaId_RESET'] = [rec_trackId2metaId[i] for i in filtered_data['metaId']]
    data = filtered_data.drop(columns=['metaId'])
    data.rename(columns={'metaId_RESET': 'metaId'}, inplace=True)
    return data


def split_fragmented(df):
    """
    寻找处分段的轨迹，并将其拆分，而后赋予新的metaID值，相当于一个trackID在同一场景下有多个meta_ID
    Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
    Formally, this is done by changing the metaId at the fragmented frame and below
    :param df: DataFrame containing trajectories
    :return: df: DataFrame containing trajectories without fragments
    """

    gb = df.groupby('metaId', as_index=False)
    # calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
    df['frame_diff'] = gb['frame'].diff().fillna(value=1.0).to_numpy()
    fragmented = df[df['frame_diff'] != 1.0]  # df containing all the first frames of fragmentation
    gb_frag = fragmented.groupby('metaId')  # helper for gb.apply
    frag_idx = fragmented.metaId.unique()  # helper for gb.apply
    df['newMetaId'] = df['metaId']  # temporary new metaId
    # 对每个metaId分组应用名为split_at_fragment_lambda的函数，该函数的作用是将轨迹在分段处进行拆分，并为拆分后的每个子轨迹分配一个新的metaId值
    df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
    # 使用factorize()方法将df数据框中的newMetaId列中的值进行编码，并将结果保存在metaId列中，同时删除newMetaId列
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    df = df.drop(columns='frame_diff')
    return df


def split_at_fragment_lambda(x, frag_idx, gb_frag):
    """ Used only for split_fragmented()
    将轨迹在分段处进行拆分，并为拆分后的每个子轨迹分配一个新的metaId值；
    该函数的输入参数x是一个数据框（DataFrame），表示按照metaId分组后的子数据框，
    frag_idx是一个包含所有发生分段的metaId值的列表，
    gb_frag是一个按照metaId分组后的GroupBy对象
    函数首先获取输入数据框中的metaId值，并初始化计数器counter为0。
    如果metaId在frag_idx列表中，说明该轨迹在分段处发生了拆分，此时需要为每个拆分出的子轨迹分配一个新的metaId值。
    函数使用gb_frag对象获取当前metaId值对应的分组索引，然后遍历这些索引，依次为每个拆分出的子轨迹分配一个新的metaId值，
    新的metaId值由原始metaId值和计数器counter组成，以metaId_counter的形式命名。最后，函数返回处理后的数据框x
     """
    metaId = x.metaId.iloc()[0]
    counter = 0
    if metaId in frag_idx:
        split_idx = gb_frag.groups[metaId]
        for split_id in split_idx:
            x.loc[split_id:, 'newMetaId'] = '{}_{}'.format(metaId, counter)
            counter += 1
    return x


def filter_short_trajectories(df, threshold):
    """
	过滤掉轨迹长度小于给定阈值threshold的轨迹--和降采样需要结合起来考虑
	Filter trajectories that are shorter in timesteps than the threshold
	:param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
	:param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
	:return: pd.df with trajectory length over threshold
	函数首先对输入数据框df按照metaId列进行分组，并统计每个分组中的数据数量，即每个轨迹的长度。
	然后，函数选择长度大于等于threshold的轨迹，并将这些轨迹对应的metaId值保存在名为idx_over_thres的变量中。
	接下来，函数从df中选择所有metaId值在idx_over_thres中出现过的行，并将结果保存在df中，以此实现轨迹长度的过滤
	"""
    len_per_id = df.groupby(by='metaId', as_index=False).count()  # sequence-length for each unique pedestrian
    idx_over_thres = len_per_id[len_per_id['frame'] >= threshold]  # rows which are above threshold
    idx_over_thres = idx_over_thres['metaId'].unique()  # only get metaIdx with sequence-length longer than threshold
    df = df[df['metaId'].isin(idx_over_thres)]  # filter df to only contain long trajectories
    return df


def groupby_sliding_window(x, window_size, stride):
    """
	对单个metaId分组的数据进行固定窗口划分
	首先计算数据的长度x_len，然后计算能够划分的窗口数量n_chunk。
	划分的方法是整体拆分，并不是按数据一个个划过去n_chunk = (x_len - window_size) // stride + 1
	接下来，函数使用一个循环遍历所有窗口，并为每个窗口中的数据分配一个新的metaId值，新的metaId值由原始metaId值和窗口编号组成，以metaId_i的形式命名。
	最后，函数从原始数据中选择所有窗口中的数据，并将新的metaId值保存到newMetaId列中，并将结果返回
	"""
    x_len = len(x)
    n_chunk = (x_len - window_size) // stride + 1
    idx = []
    metaId = []
    for i in range(n_chunk):
        idx += list(range(i * stride, i * stride + window_size))
        metaId += ['{}_{}'.format(x.metaId.unique()[0], i)] * window_size
    # temp = x.iloc()[(i * stride):(i * stride + window_size)]
    # temp['new_metaId'] = '{}_{}'.format(x.metaId.unique()[0], i)
    # df = df.append(temp, ignore_index=True)
    df = x.iloc()[idx]
    df['newMetaId'] = metaId
    return df


def sliding_window(df, window_size, stride):
    """
	Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
	chunked trajectories are overlapping
	:param df: df
	:param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
	:param stride: timesteps to move from one trajectory to the next one
	:return: df with chunked trajectories
	函数首先对数据框按照metaId列进行分组，并使用groupby_sliding_window函数对每个分组进行窗口划分。
	然后，函数使用factorize()方法将新的metaId列中的值进行编码，并将结果保存到metaId列中，同时删除newMetaId列
	"""
    gb = df.groupby(['metaId'], as_index=False)
    df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    df = df.reset_index(drop=True)
    return df

# ---------frame_ped_dict 和 frame_ped_dict_hin是可以合并的 多做处理 并无关系
def frame_ped_dict(data,data_file):
    # frame,label,x,y,sceneID,metaID => [frame,metaID,y,x,label,sceneID]
    SDD_origin_data = data.to_numpy().T
    SDD_origin_data = SDD_origin_data[[0,5,3,2,1,4], :]
    Scenelist = np.unique(SDD_origin_data[5, :]).tolist()
    all_frame_data = []
    valid_frame_data = []
    numFrame_data = []

    Pedlist_data = []
    frameped_dict = []  # peds id contained in a certain frame
    pedtrajec_dict = []  # trajectories of a certain ped
    for seti, scene in enumerate(Scenelist):
        print('preprocess  scene ' + scene + ' data')
        data = SDD_origin_data[:, SDD_origin_data[5, :] == scene]
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
                current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                # todo 添加label 和 scene
                # label = FrameContainPed[4,FrameContainPed[0,:]==frame][0]
                # scene = FrameContainPed[5,FrameContainPed[0,:]==frame][0]
                # Add their pedID, x, y to the row of the numpy array
                Trajectories.append([int(frame), current_x, current_y])
                # Trajectories.append([int(frame), current_x, current_y,label,scene])
                # 如果当前帧不在frameped_dict中，则相应的添加该帧，并将该帧包含的行人添加；记录了当前数据集的每个帧包含了那些行人
                if int(frame) not in frameped_dict[seti]:
                    frameped_dict[seti][int(frame)] = []
                frameped_dict[seti][int(frame)].append(pedi)
            pedtrajec_dict[seti][pedi] = np.array(Trajectories)
    f = open(data_file, "wb")
    # 这两个对象序列化到文件中
    pickle.dump((frameped_dict, pedtrajec_dict,Scenelist), f, protocol=2)
    f.close()


def frame_ped_dict_hin(data,data_file):
    """
    frame,label,x,y,sceneID,metaID => [frame,metaID,y,x,label,sceneID]
    返回的ped-dict中对应的添加每个行人的label，同时在此处对label进行重塑
    SDD：3类；ped = ped+cart; bike =bike+skater; car = car+bus
    """
    # 第一步 : 重塑数据的代码格式 ped = ped+cart; bike =bike+skater; car = car+bus
    # data['label'] = data['label'].replace('Cart','Pedestrian')
    # data['label'] = data['label'].replace('Skater','Biker')
    # data['label'] = data['label'].replace('Bus','Car')
    # 第一步：有两种方式 将数据视为三种格式或则仅仅只是考虑 Pededtrain，Biker 先基于该类型验证出完整代码 在后续验证时尽量运用数字 以便于通用性
    # 只保留对应的 pedestrian biker两类数据
    data = data[(data['label'] == 'Pedestrian') | (data['label'] == 'Biker')]
    # 第二步：提取frame-dict与ped-dict
    SDD_origin_data = data.to_numpy().T
    SDD_origin_data = SDD_origin_data[[0,5,3,2,1,4], :]
    Scenelist = np.unique(SDD_origin_data[5, :]).tolist()
    all_frame_data = []
    valid_frame_data = []
    numFrame_data = []
    Pedlist_data = []
    frameped_dict = []  # peds id contained in a certain frame
    pedtrajec_dict = []
    pedlabel_dict = []# trajectories of a certain ped
    for seti, scene in enumerate(Scenelist):
        print('preprocess  scene ' + scene + ' data')
        data = SDD_origin_data[:, SDD_origin_data[5, :] == scene]
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
        pedlabel_dict.append({})
        for ind, pedi in enumerate(Pedlist):
            if ind % 100 == 0:
                print(ind, len(Pedlist))
            # Extract trajectories of one person 抽取单人的轨迹数据
            FrameContainPed = data[:, data[1, :] == pedi]
            # Extract ped label
            Label = FrameContainPed[4,0]
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
                current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                # todo 添加label 和 scene
                # label = FrameContainPed[4,FrameContainPed[0,:]==frame][0]
                # scene = FrameContainPed[5,FrameContainPed[0,:]==frame][0]
                # Add their pedID, x, y to the row of the numpy array
                Trajectories.append([int(frame), current_x, current_y])
                # Trajectories.append([int(frame), current_x, current_y,label,scene])
                # 如果当前帧不在frameped_dict中，则相应的添加该帧，并将该帧包含的行人添加；记录了当前数据集的每个帧包含了那些行人
                if int(frame) not in frameped_dict[seti]:
                    frameped_dict[seti][int(frame)] = []
                frameped_dict[seti][int(frame)].append(pedi)
            pedtrajec_dict[seti][pedi] = np.array(Trajectories)
            # 保存对应的行人label维度
            pedlabel_dict[seti][pedi] = Label

    f = open(data_file, "wb")
    # 这两个对象序列化到文件中
    pickle.dump((frameped_dict, pedtrajec_dict,Scenelist,pedlabel_dict), f, protocol=2)
    f.close()


def SDD_process_data_calcuate():
    DATA_PATH = '../data/SDD/sdd_full_data.pkl'
    sdd_data = pd.read_pickle(DATA_PATH)
    # 循环分析 每个场景的数据
    grouped_scene = sdd_data.groupby('scene')
    counts_all = {}
    labels = ['Pedestrian', 'Biker','Skater', 'Cart',  'Car', 'Bus']
    for scene_name, scene_data in grouped_scene:
        # 统计场景中可以形成的序列数
        print('calcuate '+ scene_name + 'data')
        grouped_sceneId = scene_data.groupby('sceneId')
        # 计算Nos
        scene_sequence_group = 0
        # 计算NoA
        scene_agent_num = [0 for _ in range(6)]
        # 计算AV,AA
        scene_speed, scene_acceleration = [], []
        # todo 后续分析一下对应的little 68.72 过大，nexus 车辆很多的情况下 AV应该较大
        for scene_id_name,scene_id_data in grouped_sceneId:
            columns_counts = scene_id_data.nunique()
            frame_array = scene_id_data['frame'].unique()
            # 依据已有的frame列表进行分析，只有完整的20帧才被保存
            sorted_array = np.sort(frame_array)
            count = 0  # 记录符合条件的组合数量
            n = len(sorted_array)
            # 计算相应的agent数量
            for i in range(n):
                current_count = 1  # 当前组合数量，默认为1，因为起点数字自身也算一个数字
                for j in range(i + 1, n):
                    if sorted_array[j] - sorted_array[j - 1] == 12:  # 判断连续数字之间的差是否为12
                        current_count += 1
                        if current_count >= 20:  # 如果当前组合数量大于等于20，更新符合条件的组合数量
                            # 继而统计符合group条件下的各行人数据
                            match_values = [ i for i in range(sorted_array[i],sorted_array[j],12)]
                            match_values.append(sorted_array[j])
                            selected_rows = scene_id_data[scene_id_data['frame'].isin(match_values)]
                            for idx, label in enumerate(labels):
                                scene_agent_num[idx] += \
                                selected_rows[selected_rows['label'] == label]['metaId'].unique().shape[0]
                            count += 1
                            break
                    else:
                        break
            scene_sequence_group += count
            # 计算相应的agent速度以及加速度
            scene_agent = scene_id_data.groupby('metaId')
            for scene_id_agent,scene_agent_data in scene_agent:
                # 使用欧氏距离公式来计算行人在相邻帧之间的位移
                scene_agent_data['distance'] = np.sqrt((scene_agent_data['x'].diff())**2+(scene_agent_data['y'].diff())**2)
                # 计算了帧之间的时间间隔 time，
                scene_agent_data['time'] = scene_agent_data['frame'].diff()
                # 我们将位移除以时间间隔得到速度 speed
                scene_agent_data['speed'] = scene_agent_data['distance']/(scene_agent_data['time']*0.4/12)
                # 计算速度的变化率。
                scene_agent_data['acceleration'] = scene_agent_data['speed'].diff() / (scene_agent_data['time']*0.4/12)
                #  df.loc 来将缺失的时间间隔对应的速度值设为 NaN
                scene_agent_data.loc[scene_agent_data['time'].isnull(), 'speed'] = np.nan
                scene_agent_data.loc[scene_agent_data['time'].isnull(), 'acceleration'] = np.nan
                # 得到每个agent的平均速度
                average_speed = scene_agent_data['speed'].mean()
                # 计算平均加速度
                average_acceleration = scene_agent_data['acceleration'].mean()
                scene_speed.append(average_speed)
                scene_acceleration.append(average_acceleration)
        counts_all[scene_name] = [scene_sequence_group,scene_agent_num,np.mean(scene_speed), np.mean(scene_acceleration)]
    i = 0
    agent_sum = [0,0,0,0,0,0,0,0]
    for key,value in counts_all.items():
        agent_sum[i] = sum(value[1])
        i += 1
    # 计算E-D和S-D
    statistics_list = [(key, value) for key, value in counts_all.items()]
    data_list = []
    for index,data in enumerate(statistics_list):
        data_seq = data[1][0]
        data_num = data[1][1]
        data_speed = data[1][2]
        data_acceleration = data[1][3]
        data_num.append(data_seq)
        data_num.append(data_speed)
        data_num.append(data_acceleration)
        data_list.append(data_num)
    new_data_list = [[sub_list[i] for sub_list in data_list] for i in range(len(data_list[0]))]
    # 计算极端偏差（E-D）
    ed_list = [max(sub_list) - min(sub_list) for sub_list in new_data_list]
    # 计算标准差（S-D）
    sd_list = [np.std(sub_list) for sub_list in new_data_list]
    # 开始计算行人的平均速度和平均加速度

    print('Done')
    return counts_all


def SDD_preprocess_full():
    """
    基于SDD原始全部数据 处理成标准形式 后续相应的划分基于此数据集再度产生

    """
    data_dirs = {
        'bookstore': ['./data/SDD/train/bookstore_0', './data/SDD/train/bookstore_1', './data/SDD/train/bookstore_2',
                      './data/SDD/train/bookstore_3'],
        'coupa': ['./data/SDD/train/coupa_3', './data/SDD/test/coupa_0', './data/SDD/test/coupa_1'],
        'deathCircle': ['./data/SDD/train/deathCircle_0', './data/SDD/train/deathCircle_1',
                        './data/SDD/train/deathCircle_2',
                        './data/SDD/train/deathCircle_3', './data/SDD/train/deathCircle_4'],
        'gates': ['./data/SDD/train/gates_0', './data/SDD/train/gates_1', './data/SDD/train/gates_3',
                  './data/SDD/train/gates_4', './data/SDD/train/gates_5', './data/SDD/train/gates_6',
                  './data/SDD/train/gates_7', './data/SDD/train/gates_8', './data/SDD/test/gates_2'],
        'hyang': ['./data/SDD/train/hyang_4', './data/SDD/train/hyang_5', './data/SDD/train/hyang_6',
                  './data/SDD/train/hyang_7', './data/SDD/train/hyang_9', './data/SDD/test/hyang_0',
                  './data/SDD/test/hyang_1', './data/SDD/test/hyang_3', './data/SDD/test/hyang_8'],
        'nexus': ['./data/SDD/train/nexus_0', './data/SDD/train/nexus_1', './data/SDD/train/nexus_3',
                  './data/SDD/train/nexus_4', './data/SDD/train/nexus_7', './data/SDD/train/nexus_8',
                  './data/SDD/train/nexus_9', './data/SDD/test/nexus_5', './data/SDD/test/nexus_6'],
        'little': ['./data/SDD/test/little_0', './data/SDD/test/little_1', './data/SDD/test/little_2',
                   './data/SDD/test/little_3'],
        'quad': ['./data/SDD/test/quad_0', './data/SDD/test/quad_1', './data/SDD/test/quad_2', './data/SDD/test/quad_3']
        }
    SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    SDD_skip, window_size, stride = 12, 20, 20
    # 遍历data_dirs字典的键和值
    scene_data = []
    for scene, data_paths in data_dirs.items():
        print('process' + scene)
        for data_path in data_paths:
            # scene_path = os.path.join('.' + data_path, 'annotations.txt') # 单独跑的时候需要
            scene_path = os.path.join(data_path, 'annotations.txt')   # 结合到整体代码结构中需要
            scene_df = pd.read_csv(scene_path, header=0, names=SDD_cols, delimiter=' ')
            # Calculate center point of bounding box
            scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
            scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
            scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
            # 使用drop方法，将数据框中名为xmin、xmax、ymin、ymax、occluded、generated，lost的列删除
            scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost'])
            # 提取场景名称作为sceneID列的值
            scene_id = data_path.split('/')[-1]
            scene_df['sceneId'] = scene_id
            scene_df['scene'] = scene
            # 使用了列表推导式，遍历了scene_df的sceneId和trackId两列，将它们拼接起来，并用下划线连接，形成一个新的字符串，最后将所有字符串组成一个新的列表，作为rec&trackId列的值。
            # 代码中使用了zfill()方法对trackId进行了填充，使得字符串的长度为4，这样可以保证rec&trackId列中的所有字符串长度都相同
            scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
                                       zip(scene_df.sceneId, scene_df.trackId)]
            # 最终的数据格式 【trackID，x,y,frame,label,sceneID]
            scene_data.append(scene_df)
    # 使用concat将一个列表中的多个数据框合并到一起，并重新生成索引
    scene_data = pd.concat(scene_data, ignore_index=True)
    # 创建rec-trackID2metaId的字典，用于将每个唯一的rec&trackId映射到一个唯一的metaId（整数编号）
    rec_trackId2metaId = {}
    for i, j in enumerate(scene_data['rec&trackId'].unique()):
        rec_trackId2metaId[j] = i
    scene_data['metaId'] = [rec_trackId2metaId[i] for i in scene_data['rec&trackId']]
    scene_data = scene_data.drop(columns=['rec&trackId'])
    # 切分断开的轨迹 降采样 metaID的作用 行 4189055 metaID 6730  1153个trackID
    print('切分断开的轨迹')
    data_continues = split_fragmented(scene_data)
    print('降采样')
    data_downsample = downsample_all_frame(df=data_continues, step=SDD_skip)
    print('滤除过短的轨迹')
    data_filter_short = filter_short_trajectories(data_downsample, threshold=window_size)
    print('对数据进行分组，划定时间窗口')
    data_sliding = sliding_window(data_filter_short, window_size=window_size, stride=stride)
    # trackID frame label x y sceneid scene metaID
    return data_sliding


def SDD_preprocess_multi_scene(train_data_file=None, test_data_file=None):
    SDD_skip, window_size, stride = 12, 20, 20
    # train
    print('加载训练数据')
    # train_data = load_SDD(path='../data/SDD', mode='train')
    train_data = load_SDD(path='../data/SDD', mode='train')
    # 切分断开的轨迹 降采样 metaID的作用 行 4189055 metaID 6730  1153个trackID
    print('切分断开的轨迹')
    train_data = split_fragmented(train_data)
    # 行4189055 ，metaID 7381
    # SDD_origin_data = self.downsample(df=SDD_origin_data, step=self.SDD_skip)
    print('降采样')
    train_data = downsample_all_frame(df=train_data, step=SDD_skip)
    print('滤除过短的轨迹')
    train_data = filter_short_trajectories(train_data, threshold=window_size)
    print('对数据进行分组，划定时间窗口')
    train_data = sliding_window(train_data, window_size=window_size, stride=stride)
    # test
    print('加载测试数据')
    # test_data = load_SDD(path='../data/SDD', mode='test')
    test_data = load_SDD(path='../data/SDD', mode='test')
    print('切分断开的轨迹')
    # 切分断开的轨迹 降采样 metaID的作用 行 4189055 metaID 6730  1153个trackID
    test_data = split_fragmented(test_data)
    # 行4189055 ，metaID 7381
    # SDD_origin_data = self.downsample(df=SDD_origin_data, step=self.SDD_skip)
    print('降采样')
    test_data = downsample_all_frame(df=test_data, step=SDD_skip)
    print('滤除过短的轨迹')
    test_data = filter_short_trajectories(test_data, threshold=window_size)
    print('对数据进行分组，划定时间窗口')
    test_data = sliding_window(test_data, window_size=window_size, stride=stride)
    print('load raw data finish,begin preprocess data')
    return train_data, test_data


def traject_preprocess_SDD(test_set, SDD_if_filter='False',HIN = 'False', train_data_file=None, test_data_file=None):
    print(os.getcwd())
    # 第一步 ：处理最原始的数据
    # 运用的是SDD完整的全部数据 由自己从最原始的txt生成
    DATA_PATH = '../data/SDD/sdd_full_data.pkl'
    # 运用的是PECNET带的数据
    TRAIN_DATA_PATH = '../data/SDD/sdd_train_data_mutli_scene.pkl'  #
    TEST_DATA_PATH = '../data/SDD/sdd_test_data_mutli_scene.pkl'
    if not os.path.exists(DATA_PATH):
        print('preocess data from raw SDD')
        sdd_data = SDD_preprocess_full()
        print('process finish !!')
        f = open(DATA_PATH, "wb")
        pickle.dump(sdd_data, f, protocol=2)
        f.close()
        print('Done')
    if not (os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH)):
        print('process data from raw SDD multi scene')
        sdd_train_data, sdd_test_data = SDD_preprocess_multi_scene()
        print('process finish !!')
        f_train = open(TRAIN_DATA_PATH, "wb")
        pickle.dump(sdd_train_data, f_train, protocol=2)
        f_train.close()
        f_test = open(TEST_DATA_PATH, "wb")
        pickle.dump(sdd_test_data, f_test, protocol=2)
        f_test.close()
        print('Done')
    # 第二步: 依据对应的条件划分不同的训练和测试集
    if test_set in ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'nexus', 'little', 'quad']:
        DATA_PATH = '../data/SDD/sdd_full_data.pkl'
        sdd_data = pd.read_pickle(DATA_PATH)
        sdd_test_data = sdd_data[sdd_data['scene'] == test_set]
        sdd_train_data = sdd_data[sdd_data['scene'] != test_set]
        sdd_test_data = sdd_test_data.drop(columns=['scene', 'trackId'])
        sdd_train_data = sdd_train_data.drop(columns=['scene', 'trackId'])
        # frame,label,x,y,sceneID,metaID
    elif test_set == 'sdd':
        TRAIN_DATA_PATH = '../data/SDD/sdd_train_data_mutli_scene.pkl'
        TEST_DATA_PATH = '../data/SDD/sdd_test_data_mutli_scene.pkl'
        sdd_train_data = pd.read_pickle(TRAIN_DATA_PATH)
        sdd_test_data = pd.read_pickle(TEST_DATA_PATH)
        sdd_test_data = sdd_test_data.drop(columns=['trackId'])
        sdd_train_data = sdd_train_data.drop(columns=['trackId'])
        # frame,label,x,y,sceneID,metaID
    else:
        raise NotImplementedError
    # 第三步：依据相应条件决定单行人还是全部类型数据
    if SDD_if_filter == 'True':
        # 3328 行人  行210652
        sdd_train_data = sdd_train_data[sdd_train_data['label'] == 'Pedestrian']
        sdd_test_data = sdd_test_data[sdd_test_data['label'] == 'Pedestrian']
    # 此处处理成统一的形式 [frame,metaID,y,x,trackid,sceneID]
    # 第四步：处理成对应的frame-dict以及ped-dict形式
    if HIN == 'True':
        print('process HIN data')
        print('process train data frame dict hin')
        frame_ped_dict_hin(sdd_train_data,train_data_file)
        print('process test data frame dict hin ')
        frame_ped_dict_hin(sdd_test_data,test_data_file)
    elif HIN == 'False':
        print('process train data frame dict')
        frame_ped_dict(sdd_train_data,train_data_file)
        print('process test data frame dict')
        frame_ped_dict(sdd_test_data,test_data_file)
    else:
        raise NotImplementedError
    print('Done')


class SDD_Dataloader():
    def __init__(self,args):
        # dataset SDD
        # data-dirs ： sdd-full sdd_exist
        self.args = args
        #  DATASET_NAME_TO_NUM = {'bookstore':0, 'coupa':1, 'deathCircle':2, 'gates':3, 'hyang':4, 'nexus':5, 'little':6, 'quad':7,}
        if self.args.dataset == 'SDD':
            self.args.seq_length = 20
            self.args.obs_length = 8
            self.args.pred_length = 12
            self.SDD_skip = 12
            self.args.relation_num = 3
            # self.args.neighbor_thred =
            # 可以写成对应的【形式】
        self.train_data_file = os.path.join(self.args.model_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.model_dir, "test_trajectories.cpkl")

        self.test_batch_cache = os.path.join(self.args.model_dir, "test_batch_cache.cpkl")
        self.test_batch_cache_split = os.path.join(self.args.model_dir, "test_batch_cache_0.cpkl")
        # origin mldg mvdg 最终都汇聚
        self.train_batch_cache = os.path.join(self.args.model_dir, "train_batch_cache.cpkl")
        self.train_batch_cache_split = os.path.join(self.args.model_dir, "train_batch_cache_0.cpkl")

        self.train_MLDG_batch_cache = os.path.join(self.args.model_dir, "train_MLDG_batch_cache.cpkl")
        self.train_MLDG_batch_cache_split = os.path.join(self.args.model_dir, "train_MLDG_batch_cache_0.cpkl")

        self.train_MVDG_batch_cache = os.path.join(self.args.model_dir, "train_MVDG_batch_cache.cpkl")
        self.train_MVDG_batch_cache_split = os.path.join(self.args.model_dir, "train_MVDG_batch_cache_0.cpkl")

        print("creating pre_processed data from SDD raw data")
        if not (os.path.exists(self.train_data_file) and os.path.exists(self.test_data_file)):
            traject_preprocess_SDD(test_set=self.args.test_set,SDD_if_filter=self.args.SDD_if_filter,HIN = self.args.HIN,
                                   train_data_file=self.train_data_file,test_data_file=self.test_data_file)
        print("process test data ")
        if not (os.path.exists(self.test_batch_cache) or os.path.exists(self.test_batch_cache_split)):
            self.test_frameped_dict, self.test_pedtraject_dict,self.test_scene_list,self.test_pedlabel_dict = self.load_dict(self.test_data_file)
            self.dataPreprocess_HIN('test')
        self.test_batch, self.test_batchnums = self.load_split_data(self.test_batch_cache)
        print('Total number of test batches:', self.test_batchnums)

        if self.args.stage == 'origin' and self.args.phase == 'train':
            print("Preparing origin data batches.")
            if not (os.path.exists(self.train_batch_cache) or os.path.exists(self.train_batch_cache_split)):
                self.train_frameped_dict, self.train_pedtraject_dict,self.train_scene_list,self.train_pedlabel_dict = self.load_dict(self.train_data_file)
                self.dataPreprocess_HIN('train')
            # todo 如果存在相应的split——0——1文件，则需要对应的判断并计算出有多少个这样的文件 不然每次都要处理一次 ！！
            self.trainbatch, self.train_batchnums= self.load_split_data(self.train_batch_cache)
            print('Total number of training origin batches:', self.train_batchnums)

        elif self.args.stage == 'meta'and self.args.phase == 'train':
            if not (os.path.exists(self.train_MLDG_batch_cache) or os.path.exists(self.train_MLDG_batch_cache_split)):
                print("process train meta cpkl")
                self.train_frameped_dict, self.train_pedtraject_dict, self.train_scene_list, self.train_pedlabel_dict = self.load_dict(self.train_data_file)
                self.MLDG_TASK(setname='train')
            self.train_batch_MLDG_task,self.train_batch_MLDG_tasknums = self.load_split_data(self.train_MLDG_batch_cache)
            print('Total number of training MLDG task batches :', len(self.train_batch_MLDG_task))

        elif ( self.args.stage == 'MVDG' or self.args.stage =='MVDGMLDG' ) and self.args.phase == 'train':
            if not (os.path.exists(self.train_MVDG_batch_cache) or os.path.exists(self.train_MVDG_batch_cache_split)):
                self.train_frameped_dict, self.train_pedtraject_dict, self.train_scene_list, self.train_pedlabel_dict = self.load_dict(self.train_data_file)
                self.MVDG_TASK(setname='train')
            self.train_batch_MVDG_task,self.train_batch_MVDG_tasknums = self.load_split_data(self.train_MVDG_batch_cache)
            print('Total number of training MVDG task batches :', len(self.train_batch_MVDG_task))

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)

    def MLDG_TASK(self,setname):
        # 第一步 加载对应数据集以及相应参数
        frameped_dict = self.train_frameped_dict
        pedtraject_dict = self.train_pedtraject_dict
        pedlabel_dict = self.train_pedlabel_dict
        scene_list = self.train_scene_list

        cachefile = self.train_MLDG_batch_cache

        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        trainbatch_meta = []
        trainbatchnums_meta = []

        for seti, seti_frameped_dict in enumerate(frameped_dict):
            trainbatch_meta.append({})
            # 只需要在此处将data-index按相应的场景分开即可 此处data-index传入了对应的seti 故而出来的data-index会与结果有较好的对应
            data_index = self.get_data_index_meta(seti, seti_frameped_dict, setname, ifshuffle=shuffle)
            train_index = data_index
            train_batch = self.get_seq_from_index_balance_HIN(frameped_dict, pedtraject_dict,pedlabel_dict, train_index, setname,scene_list)
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
            scene_now = scene_list[seti]
            # 从字符串"bookstore_0"中提取出"bookstore"
            scene_now = scene_now[:-2]
            for i in range(len(scene_list)):
                scene_find = scene_list[i][:-2]
                if scene_find == scene_now:
                    query_seti_id.remove(i)
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
        if batch_task_num < 10:
            f = open(cachefile, "wb")
            pickle.dump(batch_task, f, protocol=2)
            f.close()
        else:
            self.pick_split_data(batch_task,cachefile,batch_size = 10)

    def MVDG_TASK(self,setname):
        # 第一步 加载对应数据集以及相应参数
        frameped_dict = self.train_frameped_dict
        pedtraject_dict = self.train_pedtraject_dict
        pedlabel_dict = self.train_pedlabel_dict
        scene_list = self.train_scene_list

        cachefile = self.train_MVDG_batch_cache

        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        trainbatch_meta = []
        trainbatchnums_meta = []
        # 第二步 按场景分解获取对应batch数据
        for seti, seti_frameped_dict in enumerate(frameped_dict):
            trainbatch_meta.append({})
            # 只需要在此处将data-index按相应的场景分开即可
            data_index = self.get_data_index_meta(seti, seti_frameped_dict, setname, ifshuffle=shuffle)
            train_index = data_index
            train_batch = self.get_seq_from_index_balance_HIN(frameped_dict, pedtraject_dict,pedlabel_dict, train_index, setname,scene_list)
            trainbatchnums = len(train_batch)
            # list（场景号） - list（windows号） -tuple （）
            trainbatch_meta[seti] = train_batch
            trainbatchnums_meta.append(trainbatchnums)
        # 第三步 形成task-list
        task_list = []
        for seti, seti_batch_num in enumerate(trainbatchnums_meta):
            if trainbatchnums_meta[seti] == 0 or trainbatchnums_meta[seti] == []:
                continue
            query_seti_id = list(range(len(trainbatchnums_meta)))
            # 第一步依据seti以及对应的scene-list找出与set相同的场景，其他不同的加入到query——seti-id里
            scene_now = scene_list[seti]
            # 从字符串"bookstore_0"中提取出"bookstore"
            scene_now = scene_now[:-2]
            for i in range(len(scene_list)):
                scene_find = scene_list[i][:-2]
                if scene_find == scene_now:
                    query_seti_id.remove(i)
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
        print('mvdg task number : '+str(new_batch_task_list_num))
        if new_batch_task_list_num < 5 :
            f = open(cachefile, "wb")
            pickle.dump(new_batch_task_list, f, protocol=2)
            f.close()
        else:
            self.pick_split_data(new_batch_task_list,cachefile,batch_size= 5)

    def dataPreprocess_HIN(self,setname):
        """
        Function to load the process data into the Dataloader object
        # 需要为每个行人添加对应的scene-id -- pedtraject-dict内部包含相应的seti，相应的取出该值即可
        """
        if setname == 'train':
            frameped_dict = self.train_frameped_dict
            pedtraject_dict = self.train_pedtraject_dict
            pedlabel_dict = self.train_pedlabel_dict
            cachefile = self.train_batch_cache
            scene_list = self.train_scene_list
        else:
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            pedlabel_dict = self.test_pedlabel_dict
            cachefile = self.test_batch_cache
            scene_list = self.test_scene_list
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True

        print(setname)
        data_index = self.get_data_index(frameped_dict, setname, ifshuffle=shuffle)
        # data-index:各自数据集中所有帧 ID 和它们所属的数据集 ID、数字化后（打乱的）的帧 ID 存储在一个 3 x N 的 NumPy 数组
        trainbatch = self.get_seq_from_index_balance_HIN(frameped_dict, pedtraject_dict, pedlabel_dict, data_index, setname,scene_list)
        trainbatchnums = len(trainbatch)
        print(str(setname)+'batch_num:'+str(trainbatchnums))
        # todo ! 改用更节省时间的序列化 train 19772 test 10112
        #  尝试将原始数据进行分批次序列化  以及在加载的时候进行反序列化
        if trainbatchnums < 40:
            f = open(cachefile, "wb")
            pickle.dump((trainbatch, trainbatchnums), f, protocol=2)
            f.close()
        else:
            self.pick_split_data(trainbatch,cachefile,batch_size=40)
        """
        f = open(cachefile, "wb")
        pickle.dump((trainbatch, trainbatchnums), f, protocol=2)
        f.close()
        """


    def get_seq_from_index_balance_HIN(self, frameped_dict, pedtraject_dict, pedlabel_dict,data_index, setname,scene_list):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''
        batch_data_mass, batch_data, Batch_id = [], [], []
        ped_cnt,last_frame = 0,0
        type_data = []  # 新加的运用于对应type
        for i in range(data_index.shape[1]):
            '''
            仍然是以对应窗口序列划分 例如test有1443帧，则相应的可以划分处1443个时间窗口，但需要后期依据
            '''
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            cur_scene = scene_list[cur_set]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            # 计算并获取对应起始帧（子轨迹）的结束帧，由于当前的子轨迹的结束帧可能会超过数据集的范围，因此使用try-expect语句块处理这种情况
            try:
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + (self.args.seq_length - 1) * self.SDD_skip])
            except:
                continue
            # todo 合并起始与结束帧中包含的行人
            present_pedi = framestart_pedi | frameend_pedi
            # 如果起始帧与结束帧没有重复的行人id，则抛弃该子轨迹
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            one_type = []  # 将行人或车辆的标记转为对应的整数
            scene_type = []  # 记录每个行人的scene-id，以及对应的起始帧
            frame_begin = [] # 记录每个行人的起始帧
            for ped in present_pedi:
                # cur-trajec：该行人对应的子轨迹数据（可能是完整的20，也可能小于20） iffull指示其是否满，ifexistobs指示其是否存在我们要求的观测帧
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                               cur_frame, self.args.seq_length,self.SDD_skip)
                if pedlabel_dict[cur_set][ped] == "Pedestrian":
                    cur_type = 0
                elif pedlabel_dict[cur_set][ped] == "Biker":
                    cur_type = 1
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                one_type.append(cur_type)
                scene_type.append(cur_scene)
                frame_begin.append(cur_frame)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            traject_batch = np.concatenate(traject, 1)
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]
            # 该windows中的行人数量
            cur_pednum = traject_batch.shape[1]
            ped_cnt += cur_pednum
            # 组合label与相应的traject
            # traject_batch_label = (traject_label,traject_batch)
            # batch_id = (cur_set, cur_frame,)
            batch_id = (cur_scene, cur_frame,cur_pednum) # 最小单位的一个batch下的数据所处的场景和id都是相同的
            # enough people in the scene
            # batch_data.append(traject_batch_label)
            if cur_pednum >= self.args.batch_around_ped * 2:
                # too many people in current scene
                # split the scene into two batches
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_batch_data, cur_Batch_id, cur_type = [], [], []
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]], traject_batch[:, ind[cur_pednum // 2:, 0]]]
                Seq_types = [one_type[ind[:cur_pednum // 2, 0]],one_type[ind[cur_pednum // 2:, 0]]]
                for idx, sb in enumerate(Seq_batchs):
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_type.append(Seq_types[idx])
                    cur_batch_data = self.massup_batch_HIN(cur_batch_data, cur_type)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                    cur_batch_data = []
                    cur_Batch_id = []
                last_frame = i
            elif cur_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                cur_batch_data, cur_Batch_id, cur_type = [], [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_type.append(cur_type)
                cur_batch_data = self.massup_batch_HIN(cur_batch_data, cur_type)
                batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                last_frame = i
            else:  # less pedestrian numbers <64
                # accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped:
                    # enough people in the scene
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
                    type_data.append(one_type)
                    # todo 需要注意的是后续相应的异质网结构的邻接矩阵会不一样 需要特殊处理 但meatID与label一一对应 可以查询的得到
                    batch_data = self.massup_batch_HIN(batch_data, type_data)
                    if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                        batch_data_mass.append((batch_data, Batch_id,))
                    elif batch_data[4] == []:
                        print('舍弃该数值')
                    # batch_data_mass.append((batch_data, Batch_id,))
                    last_frame = i
                    batch_data = []
                    Batch_id = []
                    type_data = []
                else:
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
                    type_data.append(one_type)
        if last_frame < data_index.shape[1] - 1 and batch_pednum > 1:
            batch_data = self.massup_batch_HIN(batch_data, type_data)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数值')
            # batch_data_mass.append((batch_data, Batch_id,))
        return batch_data_mass

    def massup_batch_HIN(self, batch_data,type_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        if self.args.dataset == "ETH_UCY":
            relation_num = 1
        elif self.args.dataset == "SDD" or self.args.dataset == "NBA" or self.args.dataset == "NFL":
            relation_num = 3
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]
        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 2))
        nei_list_b = np.zeros((relation_num,self.args.seq_length,  num_Peds, num_Peds))
        nei_num_b = np.zeros((relation_num,self.args.seq_length,  num_Peds))
        num_Ped_h = 0
        batch_pednum = []
        for idx, batch in enumerate(batch_data):
            cur_type = type_data[idx]
            num_Ped = batch.shape[1]
            num_Ped_type = len(cur_type)
            if num_Ped != num_Ped_type:
                print("ped-type num wrong")
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy_HIN(batch, cur_type,relation_num)
            # 相应的将该时间窗口的数据添加进batch 按第二维度 即行人的维度 （20，num-ped，2）
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            # 拼接邻接矩阵 互不影响
            nei_list_b[:, :, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, :, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            # 指示拼接到何处了
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

    def get_social_inputs_numpy_HIN(self,  inputnodes, cur_type,relation_num):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]
        num_Ped_type = len(cur_type)
        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing
        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1
        # get relative cords, neighbor id list
        nei_list = np.zeros(( relation_num,inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros(( relation_num,inputnodes.shape[0],  num_Peds))
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for rel_id in range(relation_num):
            for pedi in range(num_Peds):
                nei_list[rel_id, :, pedi, :] = seq_list
                nei_list[rel_id, :, pedi, pedi] = 0  # person i is not the neighbor of itself
                nei_num[rel_id, :, pedi] = np.sum(nei_list[rel_id, :, pedi, :], 1)
                seqi = inputnodes[:, pedi]
                for pedj in range(num_Peds):
                    seqj = inputnodes[:, pedj]
                    # 只取其中数据大于0的，即有数据的 此处默认0是确实数据 会不会造成数据的错乱
                    select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)
                    relative_cord = seqi[select, :2] - seqj[select, :2]
                    # invalid data index
                    select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                            abs(relative_cord[:, 1]) > self.args.neighbor_thred)
                    # ！！！ 进一步筛选对应的无效邻居 invalid data index
                    if pedi != pedj:
                        if rel_id == 0:
                            if cur_type[pedi] != 0 or cur_type[pedj] != 0:
                                select_dist[:] = True
                        elif rel_id == 1:
                            if cur_type[pedi] == cur_type[pedj]:
                                select_dist[:] = True
                        elif rel_id == 2:
                            if cur_type[pedi] != 1 or cur_type[pedj] != 1:
                                select_dist[:] = True
                    # invalid data index 为True则说明相应的数据需要置0 非邻居关系  运用select来控制顺序 select-dist来控制对应的index，两者相对应则可以获取对应的数值
                    nei_num[rel_id, select, pedi] -= select_dist
                    select[select == True] = select_dist
                    nei_list[rel_id, select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num

    # ----------通用的结构-----------
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

    def get_train_batch(self, idx):
        batch_data, batch_id = self.train_batch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)

        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.test_batch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False)
        return batch_data, batch_id

if __name__ == '__main__':
    print(os.getcwd())
    DATA_PATH = '../data/SDD/sdd_full_data.pkl'
    TRAIN_DATA_PATH = '../data/SDD/sdd_train_data_mutli_scene.pkl'
    TEST_DATA_PATH = '../data/SDD/sdd_test_data_mutli_scene.pkl'
    if not os.path.exists(DATA_PATH):
        print('preocess data from raw SDD')
        sdd_data = SDD_preprocess_full()
        print('process finish !!')
        f = open(DATA_PATH, "wb")
        pickle.dump(sdd_data, f, protocol=2)
        f.close()
        print('Done')
    if not (os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH)):
        print('preocess data from raw SDD multi scene')
        sdd_train_data, sdd_test_data = SDD_preprocess_multi_scene()
        print('process finish !!')
        f_train = open(TRAIN_DATA_PATH, "wb")
        pickle.dump(sdd_train_data, f_train, protocol=2)
        f_train.close()
        f_test = open(TEST_DATA_PATH, "wb")
        pickle.dump(sdd_test_data, f_test, protocol=2)
        f_test.close()
        print('Done')
    # traject_preprocess_SDD(test_set='SDD')
    #SDD_process_data_calcuate()
    SDD_traj_vis()



# 旧的原始的写在utils中的数据处理文件 traject_preprocess_SDD
"""
# -------------------SDD专门处理 基于最原始的SDD数据--------------------
def traject_preprocess_SDD(self, setname):
    '''
    data_dirs : List of directories where raw data resides
    data_file : The file into which all the pre-processed data needs to be stored
    '''
    if self.args.SDD_from == 'sdd_origin':
        if setname == 'train':
            SDD_origin_data = load_SDD(path='./data/SDD/', mode='train')
            data_file = self.train_data_file
        else:
            SDD_origin_data = load_SDD(path='./data/SDD/', mode='test')
            data_file = self.test_data_file
        print('load raw data finish,begin preprocess data')
        # 切分断开的轨迹 降采样 metaID的作用 行 4189055 metaID 6730  1153个trackID
        SDD_origin_data = split_fragmented(SDD_origin_data)
        # 行4189055 ，metaID 7381
        # SDD_origin_data = self.downsample(df=SDD_origin_data, step=self.SDD_skip)
        SDD_origin_data = downsample_all_frame(df=SDD_origin_data, step=self.SDD_skip)
        # 行349620 metaID 7381
        # 依据SDD_origin_data的scene标签 取出对应的数据data——pd [track-id,frame,label,x,y,sceneID，metaID,frame_diff]
        SDD_origin_data = SDD_origin_data.drop(columns='trackId')
        # [frame,label,x,y,sceneID，metaID] 用新的metaID 代替 trackID
        # todo 筛选出单独的行人 进行测试SDD效果
        if self.args.SDD_if_filter == 'True':
            # 3328 行人  行210652
            SDD_origin_data = SDD_origin_data[SDD_origin_data['label'] == 'Pedestrian']
    elif self.args.SDD_from == 'sdd_exist':
        # 基于PECNet新添加的数据 保证数据的统一评价标准
        if setname == 'train':
            SDD_origin_data = pd.read_pickle('./data/SDD_YNET/train_trajnet.pkl')
            data_file = self.train_data_file
            # -----------------
            colunm_counts = SDD_origin_data.nunique() 
            frame 1212 trackID 1050 x9432 y 10317 scenenID 30 metaID 8494
            0:212  12:218  24:222  36:225
            frame_counts = SDD_origin_data['frame'].value_counts()
            SDD_0 = SDD_origin_data[SDD_origin_data['frame']==0]
            SDD_0_book3 = SDD_0[SDD_0['sceneId']=='bookstore_3']
            # ------------------
        else:
            SDD_origin_data = pd.read_pickle('./data/SDD_YNET/test_trajnet.pkl')
            data_file = self.test_data_file
        print('load exist data finish,begin preprocess data')
    SDD_origin_data = SDD_origin_data.to_numpy().T
    # origin [frame,metaID,y,x,label,sceneID] 【6,193397】 exist [frame,metaID,y,x,trackid,sceneID]
    SDD_origin_data = SDD_origin_data[[0, 5, 3, 2, 1, 4], :]
    Scenelist = np.unique(SDD_origin_data[5, :]).tolist()

    all_frame_data = []
    valid_frame_data = []
    numFrame_data = []

    Pedlist_data = []
    frameped_dict = []  # peds id contained in a certain frame
    pedtrajec_dict = []  # trajectories of a certain ped
    for seti, scene in enumerate(Scenelist):
        print('preprocess  scene ' + scene + ' data')
        data = SDD_origin_data[:, SDD_origin_data[5, :] == scene]
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
                current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                # todo 添加label 和 scene
                # label = FrameContainPed[4,FrameContainPed[0,:]==frame][0]
                # scene = FrameContainPed[5,FrameContainPed[0,:]==frame][0]
                # Add their pedID, x, y to the row of the numpy array
                Trajectories.append([int(frame), current_x, current_y])
                # Trajectories.append([int(frame), current_x, current_y,label,scene])
                # 如果当前帧不在frameped_dict中，则相应的添加该帧，并将该帧包含的行人添加；记录了当前数据集的每个帧包含了那些行人
                if int(frame) not in frameped_dict[seti]:
                    frameped_dict[seti][int(frame)] = []
                frameped_dict[seti][int(frame)].append(pedi)
            pedtrajec_dict[seti][pedi] = np.array(Trajectories)
    f = open(data_file, "wb")
    # 这两个对象序列化到文件中
    pickle.dump((frameped_dict, pedtrajec_dict), f, protocol=2)
    f.close()
"""
# 原始写的一种relation-matrix的邻接矩阵以及mass-up-batch的写法
"""
    def massup_batch_HIN(self, batch_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        if self.args.dataset == "ETH_UCY":
            relation_num = 1
        elif self.args.dataset == "SDD" or self.args.dataset == "NBA" or self.args.dataset == "NFL":
            relation_num = 3

        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch[1].shape[1]
        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 2))
        nei_list_b = np.zeros((self.args.seq_length, self.args.relation_num, num_Peds, num_Peds))
        nei_num_b = np.zeros((self.args.seq_length, self.args.relation_num, num_Peds))
        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            batch_label = batch[0]
            batch_traj = batch[1]
            num_Ped = batch_traj.shape[1]
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy_HIN(batch)
            nodes_batch_b = np.append(nodes_batch_b, batch_traj, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, :, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, :, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
        
     def get_social_inputs_numpy_HIN(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        inputnodes_label = inputnodes[0]
        inputnodes_traj = inputnodes[1]
        num_Peds = inputnodes_traj.shape[1]
        relation_num = self.args.relation_num
        relation_matrix = np.zeros((num_Peds, num_Peds))
        for i in range(num_Peds):
            for j in range(num_Peds):
                if inputnodes_label[i] == 'Pedestrian' and inputnodes_label[j] == 'Pedestrian':
                    relation_matrix[i][j] = 0
                elif inputnodes_label[i] == 'Biker' and inputnodes_label[j] == 'Biker':
                    relation_matrix[i][j] = 1
                elif inputnodes_label[i] == 'Car' and inputnodes_label[j] == 'Car':
                    relation_matrix[i][j] = 2
                elif inputnodes_label[i] == 'Biker' and inputnodes_label[j] == 'Car':
                    relation_matrix[i][j] = 3
                elif inputnodes_label[i] == 'Biker' and inputnodes_label[j] == 'Pedestrian':
                    relation_matrix[i][j] = 4
                elif inputnodes_label[i] == 'Car' and inputnodes_label[j] == 'Pedestrian':
                    relation_matrix[i][j] = 5
        seq_list = np.zeros((inputnodes_traj.shape[0], num_Peds))
        # denote where data not missing
        for pedi in range(num_Peds):
            seq = inputnodes_traj[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1
        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes_traj.shape[0], relation_num, num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes_traj.shape[0], relation_num, num_Peds))

        for relation in range(relation_num):
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
            cur_nei_list = np.zeros((inputnodes_traj.shape[0],num_Peds, num_Peds))
            cur_nei_num = np.zeros((inputnodes_traj.shape[0],num_Peds))
            for pedi in range(num_Peds):
                cur_nei_list[:, pedi, :] = seq_list
                cur_nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
                cur_nei_num[:, pedi] = np.sum(cur_nei_list[:, pedi, :], 1)
                seqi = inputnodes_traj[:, pedi]
                for pedj in range(num_Peds):
                    seqj = inputnodes_traj[:, pedj]
                    select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                    relative_cord = seqi[select, :2] - seqj[select, :2]
                    # todo neighbor_thred 有问题
                    # invalid data index  添加一条，只保存对应类型的行人关系图邻居 根据关系类型来选择性地保存特定类型的行人邻居关系
                    select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                            abs(relative_cord[:, 1]) > self.args.neighbor_thred) | (
                                              relation_matrix[pedi][pedj] != relation and pedi != pedj)
                    # 自己与自己需要删除
                    # 将select_dist中为True的位置对应的邻居数量减去1，表示删除了一位邻居。
                    cur_nei_num[select, pedi] -= select_dist
                    # 将select中为True的位置根据select_dist中的值进行更新，即将需要删除的邻居的位置设置为True。
                    select[select == True] = select_dist
                    # 将cur_nei_list中需要删除的邻居位置的值设置为0，表示删除了该邻居
                    cur_nei_list[select, pedi, pedj] = 0
            nei_list[:, relation, :, :] = cur_nei_list
            nei_num[:, relation, :] = cur_nei_num
        return seq_list, nei_list, nei_num

"""