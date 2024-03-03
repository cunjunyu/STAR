import os, time, glob, re
from datetime import datetime
from tqdm import tqdm
import math
import numpy as np
from scipy import interpolate
import pandas as pd
import pickle
import random
import json
from collections import Counter
import pandas as pd
from DataProcessor.DataProcessorFactory import DatasetProcessor_BASE

"""
    数据集分析：
    FVessel 基准数据集用于评估 AIS 和视频数据融合算法的可靠性，主要包含 26 段视频及其对应的 AIS 数据，
    这些数据由位于长江武汉段的海康威视 DS-2DC4423IW-D 球机和赛阳 AIS9000-08 B 类 AIS 接收器捕获。
    为了保护隐私，数据集中每艘船的 MMSI 已被随机数字替换。
    如图 1 所示，这些视频在许多地点（例如，桥梁区域和河边）和各种天气条件下（例如，晴天、多云和低光照）拍摄。
    1.实验设置：使用过去10个点的轨迹点预测未来的5个轨迹点 
    2：AIS频率：
        类A AIS设备在正常条件下，船舶航速超过3节时，每2到10秒发送一次动态信息。
        如果船速低于3节，动态信息的发送频率减少到每3分钟一次。
        转弯时，动态信息的发送频率增加到每2秒一次。
        =================
        类B AIS设备的数据分布频率通常低于类A设备，可能每30秒到几分钟发送一次
        在繁忙的航道或临近港口时，AIS设备可能会设置为以更高的频率发送数据，以确保航行安全和避免碰撞。在较为开阔的海域，发送频率可能会较低。
    3：
    

"""
def normalization(data=None, min_=None, max_=None):
    """
    对传入参数列表进行归一化操作
    :param max_:
    :param min_:
    :param data:
    :return:
    """
    data = float(data)
    new_a = (data - min_) / (max_ - min_)
    return new_a

def time2stamp(Time):
    """
    于将一个包含年、月、日、小时、分钟、秒和毫秒的时间元组转换为时间戳（毫秒级别）和特定格式的字符串

    """
    name = "%d_%02d_%02d_%02d_%02d_%02d_%03d"%(Time[0],Time[1],Time[2],Time[3],Time[4],Time[5],Time[6])
    datetime_obj = datetime.strptime(name, "%Y_%m_%d_%H_%M_%S_%f")
    timeStamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return timeStamp, name

def get_speed(timeA, lonA, latA, timeB, lonB, latB):
    """
    传入两点的时间和经纬度信息
    时间为13位时间戳信息：ms→h
    :param timeA:
    :param lonA:
    :param latA:
    :param timeB:
    :param lonB:
    :param latB:
    :return:
    """
    a_time = int(timeA)
    b_time = int(timeB)
    a_lon = float(lonA)
    a_lat = float(latA)
    b_lon = float(lonB)
    b_lat = float(latB)
    d_timestamp = (b_time - a_time) / 1000 / 3600
    distance = get_distance(a_lon, a_lat, b_lon, b_lat)
    speed = distance / d_timestamp  # 获取平均速度，单位 Km/h
    # 千米每小时转化为米每秒
    return speed / 36

def get_course(lonA, latA, lonB, latB):
    """
    point p1(latA, lonA)
    point p2(latB, lonB)
    根据两点经纬度计算方向角，默认北半球
    :param latA:
    :param lonA:
    :param latB:
    :param lonB:
    :return:
    """
    radLonA = math.radians(lonA)
    radLatA = math.radians(latA)
    radLonB = math.radians(lonB)
    radLatB = math.radians(latB)
    dLon = radLonB - radLonA
    y = math.sin(dLon) * math.cos(radLatB)
    x = math.cos(radLatA) * math.sin(radLatB) - math.sin(radLatA) * math.cos(radLatB) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    brng = (brng + 360) % 360
    return (round(brng) / 360) * math.pi

def get_distance(lon1, lat1, lon2, lat2):
    """
    计算经纬度之间的距离，单位千米
    :param lon1: A点的经度
    :param lat1: A点的纬度
    :param lon2: B点的经度
    :param lat2: B点的纬度
    :return:
    """
    # 地球半径
    EARTH_RADIUS = 6378.137
    EARTH_RADIUS = 6378.137
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = c * EARTH_RADIUS
    return distance

def get_derivative(f, delta=1e-10):
    """导函数生成器"""

    def derivative(x):
        """导函数"""
        return (f(x + delta) - f(x)) / delta

    return derivative

def FVessel_data_load(file_save_path =None):
    """
    0：从最原始的数据源中读取数据 存为dataframe！
    // 不进行滤波跳值处理 故而
    mmsi	lon	lat	speed	course	heading	type	timestamp
    # 对timestamp进行处理 只保留秒级数据

    """
    # 字符串前的 r 表示这是一个原始字符串
    # 指定保存文件的路径和文件名
    output_file_path = file_save_path
    base_dir = r'./data/ship_FVessel'
    # 逐个读取文件夹
    # Using os.listdir() to list all entries in 'base_dir' and os.path.isdir() to check if they are directories
    video_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    # Dictionary to hold the dataframes
    dataframes = {}
    # Iterate over the folder names
    for video_folder in video_folders:
        print("process"+video_folder+"data")
        # Construct the path to the AIS folder for the current video folder
        ais_folder_path = os.path.join(base_dir, video_folder, 'ais')
        # Initialize a list to collect dataframes for each video folder
        video_dataframes = []
        # List all the CSV files in the AIS folder
        csv_files = [f for f in os.listdir(ais_folder_path) if f.endswith('.csv')]
        # Read each CSV file and store the dataframe in the list
        for csv_file in tqdm(csv_files,desc="读取该文件夹下的csv数据"+str(len(csv_files)),unit="item"):
            # Construct the full path to the CSV file
            file_path = os.path.join(ais_folder_path, csv_file)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            # Add the dataframe to the list
            video_dataframes.append(df)
        # Combine all the dataframes for the current video folder into a single dataframe
        combined_df = pd.concat(video_dataframes, ignore_index=True)
        # 按时间序列重新排序
        # Add the combined dataframe to the dictionary with the key as the video folder name
        dataframes[video_folder] = combined_df
    print("finish all FVeseel data")
    # 保存数据
    # 创建一个空的NumPy字典，用于存储DataFrame数据
    data_dict = {}
    # 将每个DataFrame的数据存储到NumPy字典中
    for scenario_name, df in dataframes.items():
        data_dict[scenario_name] = df.to_numpy()
        # 将一个13位时间戳的数据截断到秒级别，从而去除毫秒部分 将13位时间戳除以1000来实现，将其转换为以秒为单位的时间戳
        # 从数组中提取最后一列的时间戳
        timestamps = data_dict[scenario_name][:, -1]
        # 将13位时间戳截断到秒级别
        timestamps_seconds = (timestamps // 1000)*1000
        # 如果需要，将截断后的时间戳替换回原数组的最后一列
        data_dict[scenario_name][:, -1] = timestamps_seconds
    # 保存数据到.npz文件
    np.savez(output_file_path, **data_dict,allow_pickle=True)
    print("save all data!")

def clear_list(gjd_list=None):
    """
    1. 删除时间戳重复的数据，保留时间戳第一次出现的数据
    2. 跳点处理：
        (1) 在下一轨迹点与当前轨迹点的平均速度超过阈值时，按跳点处理，默认删除
        (2) 相邻轨迹点时间差大于阈值(6h)，分割列表

    :param time_interval: 时间分割阈值，相邻两点超过特定值，此处为5分钟，以此分割轨迹段
    :param max_speed: 正常速度阈值，相邻两点平均速度超过设定值(此处为100Km/h)，删除异常速度的轨迹点
        内河上的船舶速度通常比开放海洋上的速度较低，高速：船舶速度超过10节。这可能适用于某些内河巡逻船、高速客船或运输船。
        如果船舶速度为10节，那么它的速度约为 10 x 1.852 = 18.52 km/h；故而将超过30km/h的视为错误的
    :param gjd_list: 轨迹点列表
    :return: data_list
    """
    # todo 超参数 ！！！！ 决定了数据量值
    # SOG速度阈值(单位Km/h)
    max_speed = 100 # 原100 SOG_threshold
    # 相邻时间间隔超过阈值6mins时，以此切分轨迹段 6/60 = 0.1h
    time_interval = 0.5 # 原 1 // 似乎是合适的 todo 其实需要斟酌 这些数值很不一样
    if not gjd_list:
        return '传入列表为空'
    # ==================================
    # 按照时间去重，存放所有时间戳索引的列表
    timestamp_list = []
    # 重复时间的索引列表，保留出现的第一个时间戳，把后续的删除(暂时删除，输出出来)
    repeat_timestamp = []
    for gjd_index, gjd in enumerate(gjd_list):
        # 拿到每个轨迹点信息和对应索引，进行分别处理
        gjd_timestamp = gjd[-1]  # 对应轨迹点的时间戳
        if gjd_timestamp in timestamp_list:  # 如果当前时间戳已经在列表中存在，则该索引以及前一个索引对应的时间戳重复
            # print(gjd_index, '与上一个重复,将索引加入到列表中')
            repeat_timestamp.append(gjd_index)
            # 后续考虑是否删除  repeat_timestamp 为删除做准备
            # print('删除', gjd_index, '的轨迹点')
        timestamp_list.append(gjd_timestamp)  # 每个时间戳放到列表中，可以判断是否存在重复
    # 删除重复轨迹点
    normal_time_list = [gjd_list[i] for i in range(len(gjd_list)) if (i not in repeat_timestamp)]  # 包含正常时间的轨迹点
    # =================================
    # 处理跳点(按照相邻两点的平均速度) 分割轨迹点
    current_index = 0
    running_periods = []
    current_running_period = []
    data_list = []
    while current_index < len(normal_time_list):
        current_point = normal_time_list[current_index]  # 当前索引对应的轨迹点
        lon = float(current_point[1])
        lat = float(current_point[2])
        timestamp = int(current_point[-1])
        next_index = current_index + 1
        if next_index < len(normal_time_list):
            next_point = normal_time_list[next_index]  # 当前点的下一轨迹点
            next_lon = float(next_point[1])
            next_lat = float(next_point[2])
            next_timestamp = int(next_point[-1])
            distance = get_distance(lon, lat, next_lon, next_lat)  # 单位千米
            # 获取两点之间的时间差，将ms转换为h
            d_timestamp = (next_timestamp - timestamp) / 1000 / 3600
            # 如果d_timestamp!=0 d_timestamp=d_timestamp，如果=0则赋值为0.000001
            d_timestamp = d_timestamp if d_timestamp else 0.0000001
            # 如果时间差超过设定阈值，按照时间分割轨迹段
            if d_timestamp >= time_interval:
                print("时间差"+str(d_timestamp)+"超过设定阈值，按照时间分割轨迹段")
                current_running_period.append(normal_time_list[current_index])
                running_periods.append(current_running_period)
                current_running_period = []
                current_index += 1
                continue
            avg_speed = distance / d_timestamp  # 获取平均速度，单位 Km/h
            # 如果平均速度超过阈值，判断为跳点，跳过跳点，将当前轨迹点加入到航行段中
            if avg_speed >= max_speed:
                # 阈值设高一点 注意此处的ais频率不一样的会体现出来 ！
                print("平均速度"+str(avg_speed)+"超过阈值，判断为跳点，跳过跳点")
                print("distance:"+str(distance)+"  d_timestamp:"+str(d_timestamp))
                current_running_period.append(normal_time_list[current_index])
                current_index += 2
                continue
            else:
                current_running_period.append(normal_time_list[current_index])
                current_index += 1
        # 如果循环已经到达最后一个点，则直接将当前轨迹点添加
        elif next_index == len(normal_time_list):
            current_running_period.append(normal_time_list[current_index])
            running_periods.append(current_running_period)
            break
        # 分割后若每条轨迹点大于10点，则当作有效轨迹保存
    for running_period in running_periods:
        if len(running_period) > 10:
            data_list.append(running_period)

    return data_list

def list_interpolation(list_datas=None):
    """
    对时间不均的轨迹点列表进行插值，按照设定的时间间隔进行插值
    插值函数选择三次样条函数
    :param list_datas: 轨迹点列表
    :param equal_time_interval:
    :param last_time_interval:
    :return: 返回插值好的新的轨迹点列表  new_times[i], new_lat_list[i], new_lon_list[i]

    """
    # 定义等时间差阈值为10s, 由于是13位时间戳，定义间隔10s 需要10*1000ms
    # 13位时间戳指的是以毫秒为单位的 Unix 时间戳
    equal_time_interval = 1000*10 # 原海洋 60 * 1000 * 10
    # 定义最后一个等间隔时间戳与真实时间戳的差值阈值
    last_time_interval = 1000 * 10 * 1.5 # 原海洋 100 * 1000 * 10
    # 具有不均匀时间间隔 （running_period） 的轨迹点列表
    running_period = list_datas
    total_running_periods = []  # 对应船只航行段的总量列表
    new_running_period = []  # 新的航行段的信息
    timestamp_list = []  # 当作插值函数自变量
    new_timestamp_list = []  # 等时间间隔的时间戳列表
    lat_variable_list = []
    lon_variable_list = []
    variable_list = [lat_variable_list, lon_variable_list]  # 存放原始数据的因变量，暂定经纬度，需要通过插值获得
    # 将原始的字段对应的值加到原始因变量列表中
    for running_gjd in running_period:
        timestamp = int(running_gjd[-1])
        timestamp_list.append(timestamp)  # 自变量列表
        lat = float(running_gjd[2])
        lat_variable_list.append(lat)
        lon = float(running_gjd[1])
        lon_variable_list.append(lon)
    first_timestamp = timestamp_list[0]  # 真实轨迹点第一个时间戳，也是插值后的第一个轨迹点的真实时间戳
    last_timestamp = timestamp_list[-1]  # 真实轨迹点最后一个时间戳
    # 它通过计算最后一个时间戳和第一个时间戳之间的时间差，并将其转换为秒，然后除以等时间间隔来估计循环次数
    range_number = int((last_timestamp - first_timestamp) / 1000) + 3  # 获得大约循环次数数值
    # 按照第一个时间戳，进行后面等时间间隔的累加，加到与最后一个时间戳的差值在阈值范围
    for i in range(range_number):
        # todo 生成一个从开始到结尾的完整的轨迹序列点 但需要注意的是 不同的mmsi开始的轨迹点不一样 导致这处的选取可能不一样
        set_timestamp = first_timestamp + i * equal_time_interval
        new_timestamp_list.append(set_timestamp)
        if set_timestamp > last_timestamp and abs(set_timestamp - last_timestamp) < last_time_interval:
            # print(set_timestamp)
            break
    # print('----------------------------------')
    # 将时间戳列表减去第一个时间戳并除以1000 转换为函数可以接受的小范围(13位时间戳作为自变量组成的函数值超出浮点数范围)
    origin_independent_variable_list = []
    new_independent_variable_list = []
    for timestamps_0 in timestamp_list:
        origin_time = (timestamps_0 - first_timestamp) / 1000  # 转换成小数据格式
        origin_independent_variable_list.append(origin_time)
    for timestamps_1 in new_timestamp_list:
        new_time = (timestamps_1 - first_timestamp) / 1000  # 转换成小数据格式
        new_independent_variable_list.append(new_time)
    # 将时间自变量列表转换成数组，方便后续三次样条插值的输入
    origin_independent_variable_list = pd.Series(origin_independent_variable_list)
    new_independent_variable_list = pd.Series(new_independent_variable_list)
    # 插值法获得的新的因变量列表(lat,lon,sog,cog)
    new_lat_list = []
    new_lon_list = []
    # 获得对应索引的拉格朗日插值函数，按照列表variable_list所示0-lat,1-lon,2-sog,3-cog
    for index, dependent_variable in enumerate(variable_list):
        dependent_variable = pd.Series(dependent_variable)  # 样条函数
        fx = interpolate.splrep(origin_independent_variable_list, dependent_variable)  # 三次样条函数插值
        yy = interpolate.splev(new_independent_variable_list, fx, der=0)  # 获得对应新时间戳的自变量数组列表
        for i in yy:
            y_ = np.around(i, 15)  # 对numpy中的float64类型保留5位有效数字-->15
            if index == 0:
                new_lat_list.append(y_)
            elif index == 1:
                new_lon_list.append(y_)
    new_times = []
    # 转换回对应的13位格式
    for new_time_ in new_independent_variable_list:
        new_time_ = new_time_ * 1000 + first_timestamp
        new_times.append(int(new_time_))
    # total_running_periods.append(new_running_period)
    for i in range(len(new_times)):
        new_list = [new_times[i], new_lat_list[i], new_lon_list[i]]
        total_running_periods.append(new_list)
    return total_running_periods

def list_segmentation(all_gjd_list=None):
    """
    对包含轨迹点的列表进行分割
    parking_dis=stop_DISTANCE, parking_length=parking_number
    :param parking_length:
    :param parking_dis:
    :param all_gjd_list:
    :return:
    """
    # 此处删除了很多的数据 基本上皆是此处删除的 ！！ 动与不动 是否很重要 或则说是否确实太多的不动了 或则阈值设置的问题
    # 慢速为2节 其即为3.6km/h；故而相应的10s约10m，符合运行速度 再小可以视为未动
    # 判定行驶到停留点的阈值，插值后的相邻两点距离小于10m，判断为行驶到停留点，当相邻两点超过10m，为航行状态，单位Km
    parking_dis = 10 * 0.001 # 海洋上  100 * 0.001 100m =>对应的人步行速度约为1.1-1.5m/s 则10s内超过10m方可！！
    # 连续三个停留点在停驻段才算停留，小于三合并到航行段==》提高停驻段数据量阈值 ！！
    parking_length = 5
    parking_periods = []
    parking_index = []  # 停驻段索引列表
    for current_index, continuous_gjd in enumerate(all_gjd_list):
        current_lat = continuous_gjd[2]
        current_lon = continuous_gjd[1]
        next_index = current_index + 1
        if next_index < len(all_gjd_list):
            next_lat = all_gjd_list[next_index][2]
            next_lon = all_gjd_list[next_index][1]
            distance = get_distance(current_lon, current_lat, next_lon, next_lat)
            if distance <= parking_dis:  # 进入停驻段
                parking_index.append(current_index)
            else:  # 驶出停驻段，进入新的航行段

                if parking_index:
                    parking_periods.append(parking_index)
                    parking_index = []
                else:
                    continue
        elif next_index == len(all_gjd_list):
            if parking_index:
                parking_periods.append(parking_index)
            else:
                continue
    # print(1)
    running_periods = []  # 存放航行轨迹段
    running_index = []  # 存放每个航行轨迹段的始终索引
    # 通过停驻段判断航行段
    for index, parking_period in enumerate(parking_periods):
        # 遍历停驻段列表，如果当前停驻段列表长度大于设定阈值，判断为船舶停止状态，以此切断分割航行段
        if len(parking_period) > parking_length:
            # 单独判断第一个停驻段列表，观察全部轨迹点是以行驶状态开始还是停驻状态开始
            if index == 0:
                if parking_period[0] == 0:
                    # 如果从第一个轨迹点开始航速就为0，第一段航行段起始点为停驻段后一个点
                    running_start_index = parking_period[-1] + 1
                    running_index.append(running_start_index)  # 将航行段起始索引加入到航行轨迹段的索引列表
                else:
                    # 如果第一个轨迹点开始航速不为0，航行段索引值从0开始，以第一个停驻段前一个索引值为结束
                    running_start_index = 0
                    running_end_index = parking_period[0] - 1
                    running_index = [running_start_index, running_end_index]  # 得到一条航行轨迹段，根据索引值确定
                    running_periods.append(running_index)  # 将该航行段加入到航行段列表中
                    running_index = []  # 航行段索引列表清空，预计存放下一个航行段
            elif running_index:
                # 如果航行段索引列表存在，说明只有航行段起始索引，填入航行段结束索引
                running_end_index = parking_period[0] - 1
                running_index.append(running_end_index)
                running_periods.append(running_index)
                running_index = []
            else:
                # 航行段索引为空，判断当前停驻段后索引轨迹点为航行段起点，加到航向段索引列表中
                running_start_index = parking_period[-1] + 1
                running_index.append(running_start_index)
    # 如果循环停驻段结束后，航行段索引列表仍不为空，证明最后一段停驻段后仍在航行，直到整段轨迹段结束，将所有轨迹点最后一个轨迹点索引加入
    if len(running_index) == 1:
        running_index.append(len(all_gjd_list) - 1)
        running_periods.append(running_index)

    # 將航行段索引对应的轨迹点加入列表返回
    running_gjd_periods = []

    for running_index_period in running_periods:
        running_period = []
        running_gjd_period = all_gjd_list[running_index_period[0]:running_index_period[1] + 1]
        # 对按照停驻段分割的航行段按照时间间隔阈值进行分割
        for running_gjd_index, every_gjd in enumerate(running_gjd_period):
            running_period.append(every_gjd)
        running_gjd_periods.append(running_period)

    return running_gjd_periods

def clear_jump_point_by_D_value(ship_points ):
    """
    这个函数的作用是去除AIS数据中的那些可能因为错误、噪声或其他非正常航行行为而出现速度跳变的点。
    这对于数据预处理非常有用，可以提高后续分析步骤的准确性。
    =======================================
    ship_points:船舶标识(MMSI码)
    D_value:速度变化的阈值 原10--》现5
    default_speed:速度的默认阈值。如果轨迹的第一个点的速度超过这个值，它将被视为异常并删除。 原30--》现20
    ========================================
    使用for循环遍历ship_points字典中的每一条轨迹。
    对于每条轨迹，它首先检查轨迹的第一个点的速度。如果第一个点的速度超过default_speed，则删除这个点。
    这个过程将重复直到轨迹的第一个点的速度不超过default_speed。
    然后函数进入另一个while循环，从当前轨迹点开始，检查下一个点的速度是否比当前点的速度大了超过D_value。
    如果是，那么从下一个点开始到轨迹的末尾的所有点都被认为是异常的，并将被删除。这个过程会一直持续到不再发现速度异常的点。
    最终，函数返回清理后的ship_points字典。
    """
    # todo 超参数
    D_value = 5
    default_speed = 20
    for ship, trajectory_list in tqdm(ship_points.items()):
        for points in trajectory_list:

            while len(points) > 0:
                if points[0][4] > default_speed:
                    del points[0]
                else:
                    break

            current_point_index = 0
            while current_point_index < len(points):
                next_point_index = current_point_index + 1
                if next_point_index <= len(points) - 1:
                    current_speed = points[current_point_index][4]
                    next_speed = points[next_point_index][4]
                    if next_speed - current_speed > D_value:
                        del points[next_point_index:len(points)]
                        continue
                    else:
                        current_point_index += 1
                else:
                    break
    return ship_points


def FVessel_data_classifer(origin_data):
    """
    input:dataframe
    output:
    作用：
    1. 根据MMSI进行数据分类，排序
    """
    # 创建一个空字典，用于存储分组后的数据
    mmsi_data = {}
    # 删除第一列
    origin_data_without_index = np.delete(origin_data,0,axis=1)
    # 迭代数组的每一行
    for row in origin_data_without_index:
        key = row[0]  # 第一列是键
        if key not in mmsi_data:
            mmsi_data[key] = []
        mmsi_data[key].append(row[:].tolist())  # 添加剩余部分作为值
    # aggregated_data 是一个列表的列表，其中每个子列表包含具有相同第一列值的行
    # 按照时间戳进行排序
    cacluate_nums_for_mmsi = []
    for mmsi in tqdm(mmsi_data.keys()):
        vessel_type = None
        # 获取船舶的静态特征
        vessel_type = mmsi_data[mmsi][0][6]
        if not vessel_type:
            vessel_type = -1  # todo船舶类型为空时，定义编号为 100
        sorted_list = sorted(mmsi_data[mmsi], key=lambda x: x[-1])  # 按照时间戳进行排序,时间戳位于字段位置的最后一个
        # todo 不断的在进行轨迹的拆分！！ 时间阈值 速度阈值 航行段
        timed_lists = clear_list(gjd_list= sorted_list)  # 按照时间间隔阈值分段轨迹段 get: [[1],[2],[3]]
        running_period = []
        before_segmentation = 0
        middle_segmentation_interpolation = 0
        after_segmentation = 0
        # 对按照时间分割的轨迹段进行分段插值  ==> 前述的时间差值和速度max一般不会超越 故而删值多在后续内容
        for timed_list in timed_lists:
            before_segmentation += len(timed_list)
            if(len(timed_list)<20):
                print("数据点较少 需要插值分析一下")
            interpolated_list = list_interpolation(timed_list)
            # 进行轨迹分割，按照相邻轨迹点的距离小于阈值(暂定10m)进入停驻段，第一个距离大于阈值跳出停驻段
            middle_segmentation_interpolation += len(interpolated_list)
            # todo 不进行轨迹的航段和驻留段的分析 直接插值完后全部运用数据 ！！ 大概率无效果 不需要更改？
            segmented_list = list_segmentation(interpolated_list)
            # segmented_list = interpolated_list
            # 统计原本mmsi内包含的轨迹
            # print(str(mmsi) + "对应的时间段的数量为" + str(len(timed_list))+"经过轨迹分割后的为"+str(len(segmented_list)))
            for a in segmented_list:
                after_segmentation += len(a)
                running_period.append(a)
        # 根据经纬度计算航速航向，补充到轨迹点字段信息中
        cacluate_nums_for_mmsi.append([mmsi,before_segmentation,middle_segmentation_interpolation,after_segmentation])
        new_running_periods = []
        for running_period_ in running_period:  # 拿到每个航行段，计算每个gjd的速度航向
            # 航行段列表
            current_running_period = []
            # 当前时间戳对应的时间列表
            current_time_list = []
            # 累计航行距离
            all_running_distance = 0
            # 当前时间戳累计航行距离列表
            current_distance_list = [0]
            current_course = None  # 默认航向
            if len(running_period_) >= 2:  # 轨迹点为1，不能计算航向航速
                starting_point_lon_lat = [running_period_[0][2], running_period_[0][1]]
                for current_index, current_gjd in enumerate(running_period_):
                    # 先计算对应轨迹点已经行驶的距离
                    first_time = running_period_[0][0]
                    current_timestamp = current_gjd[0]  # 当前轨迹点的时间戳
                    current_time_list.append((int(current_timestamp) - int(first_time)) / 1000)  # 转换成s
                    current_lon = current_gjd[2]  # 当前轨迹点的经度
                    current_lat = current_gjd[1]  # 当前轨迹点的纬度
                    current_gjd_list = [current_timestamp, current_lat, current_lon]  # 增加字段后的轨迹点
                    next_index = current_index + 1  # 下一轨迹点的索引
                    if next_index < len(running_period_):
                        next_lon = running_period_[next_index][2]  # 下一轨迹点的经度
                        next_lat = running_period_[next_index][1]  # 下一轨迹点的纬度
                        # 算前后两点的距离  km
                        distance = get_distance(current_lon, current_lat, next_lon, next_lat)
                        all_running_distance += distance  # 加和之前的航行距离
                        current_distance_list.append(all_running_distance)  # 对应时间戳添加距离
                        current_course = get_course(current_lon, current_lat, next_lon, next_lat)  # 得到当前点和后一点的航向
                        current_gjd_list.append(current_course)
                    elif next_index == len(running_period_):  # 如果当前点是最后一个航行轨迹点
                        if current_course:
                            last_course = current_course
                        else:
                            last_course = 0
                        current_gjd_list.append(last_course)
                    #原来(时间戳,纬度,经度)-----现在current_running_period为(时间戳,纬度,经度,角度)
                    current_running_period.append(current_gjd_list)
                # print('-----------------------')
                time_variable = pd.Series(current_time_list)
                distance_variable = pd.Series(current_distance_list)
                gx = interpolate.interp1d(time_variable, distance_variable,
                                          fill_value="extrapolate")  # 插值一个一维函数 返回interp1d
                # 注*： fill_value="extrapolate" 外推到范围外，可能会有误差
                # 位移对时间求导代表速度，填入对应时间值即可获得对应的速度
                gd = get_derivative(gx)  # 对应位移时间函数的导函数，填入对应的时间值
                # 对应时间戳的速度列表
                v_list = []
                for time_point in current_time_list:
                    current_v = gd(time_point)
                    v_list.append(current_v * 1000)  # 单位km/s 转换成m/s
                new_gjd_list = []
                # 添加一系列的信息 对应的Marine的13位格式数据 ！！
                for index, gjd in enumerate(current_running_period):
                    gjd.append(round(v_list[index], 3))
                    gjd.append(round(current_distance_list[index], 5))
                    timestamp = int(current_running_period[0][0])  # 13位时间戳，转换成月日时的形式
                    data_time = time.localtime(timestamp / 1000)
                    # 轨迹点中加入类型信息
                    gjd.append(int(vessel_type))
                    # 根据日期拿到月日时
                    time_month = data_time.tm_mon
                    time_day = data_time.tm_mday
                    time_hour = data_time.tm_hour
                    gjd.append(time_month)
                    gjd.append(time_day)
                    gjd.append(time_hour)
                    # gjd.append(starting_point_lon_lat)
                    # 轨迹点中加入开始的轨迹点的经纬度
                    gjd.append(running_period_[0][2])
                    gjd.append(running_period_[0][1])
                    # 轨迹点中加入mmsi编号
                    gjd.append(int(mmsi))
                    new_gjd_list.append(gjd)
                new_running_periods.append(new_gjd_list)
        mmsi_data[mmsi] = new_running_periods

    # 根据前后两点的速度之差去除跳点：默认开始点为正常点，计算前一点与后一点的速度差值，超过阈值舍去后一点，依次循环计算
    mmsi_data = clear_jump_point_by_D_value(mmsi_data)
    # 获取经纬度最大值，最小值
    max_list = [-10000, -10000, -10000, -10000, -10000]  # lat,lon
    min_list = [100000, 110000, 110000, 110000, 110000]  # lat,lon
    for mmsi, running_periods in tqdm(mmsi_data.items()):
        for running_period in running_periods:
            for gjd in running_period:
                currt_lat = float(gjd[1])
                if currt_lat > max_list[0]:
                    max_list[0] = currt_lat
                elif currt_lat < min_list[0]:
                    min_list[0] = currt_lat
                currt_lon = float(gjd[2])
                if currt_lon > max_list[1]:
                    max_list[1] = currt_lon
                elif currt_lon < min_list[1]:
                    min_list[1] = currt_lon

                currt_cog = float(gjd[3])
                if currt_cog > max_list[2]:
                    max_list[2] = currt_cog
                elif currt_cog < min_list[2]:
                    min_list[2] = currt_cog
                currt_sog = float(gjd[4])
                if currt_sog > max_list[3]:
                    max_list[3] = currt_sog
                elif currt_sog < min_list[3]:
                    min_list[3] = currt_sog
                currt_dis = float(gjd[5])
                if currt_dis > max_list[4]:
                    max_list[4] = currt_dis
                elif currt_dis < min_list[4]:
                    min_list[4] = currt_dis
    # 进行归一化 todo 此处用的只是对应的一个视频的 故而如果进行归一化的话其实不太准确！
    # for mmsi, running_periods in tqdm(mmsi_data.items()):
    #     if running_periods:
    #         for run_period in running_periods:
    #             if len(run_period) < 15:
    #                 running_periods.remove(run_period)
    #     if not running_periods:
    #         del mmsi_data[mmsi]
    #     for running_period in running_periods:
    #         for gjd in running_period:
    #             currt_lat = float(gjd[1])
    #             currt_lon = float(gjd[2])
    #             currt_cog = float(gjd[3])
    #             currt_sog = float(gjd[4])
    #             currt_dis = float(gjd[5])
    #             normalized_lat = normalization(currt_lat, min_list[0], max_list[0])
    #             gjd[1] = normalized_lat
    #             normalized_lon = normalization(currt_lon, min_list[1], max_list[1])
    #             gjd[2] = normalized_lon
    #             normalized_cog = normalization(currt_cog, min_list[2], max_list[2])
    #             gjd[3] = normalized_cog
    #             normalized_sog = normalization(currt_sog, min_list[3], max_list[3])
    #             gjd[4] = normalized_sog
    #             normalized_dis = normalization(currt_dis, min_list[4], max_list[4])
    #             gjd[5] = normalized_dis
    # print('----------------------')
    res = {
        "ship_data":mmsi_data,
        "max_min": [max_list, min_list],
        "nums_change":cacluate_nums_for_mmsi
    }
    return res




class DatasetProcessor_ship_FVessel(DatasetProcessor_BASE):
    def __init__(self,args):
        """

        """
        # 经度在前，纬度在后（ [longitude, latitude] ）
        args.sample_num = 1
        if args.denormalize == 'True':
            args.data_dir = args.save_base_dir +  str(args.dataset)+'/' + str(args.test_set) + '/' + str(args.train_model) +'_' + str(args.stage) + str(args.denormalize)+'/'
            args.model_dir = args.save_base_dir + str(args.dataset) + '/' + str(args.test_set) + '/' + str(
                args.train_model) + '_' + str(args.stage) + str(args.denormalize) + '/' + str(args.param_diff) + '/'
            args.neighbor_thred = 0.03  # 需要调节分析
            print("模型的新保存地址在"+args.model_dir)
        else:
            args.neighbor_thred = 0.001 # 需要调节分析
        super().__init__(args=args)
        assert self.args.dataset == "Ship_FVessel"
        print("正确完成真实数据集" + self.args.dataset + "的初始化过程")
        print("经纬度的邻居判断阈值为"+str(self.args.neighbor_thred))

    def data_preprocess_for_origintrajectory(self, args):
        """
        0：从最原始的数据源中读取数据 存为dataframe！
        1. 根据MMSI进行数据分类，排序
        2. 对每个MMSI船舶的轨迹点列表进行清洗，删除重复时间戳的数据，删除跳点数据
        3. 按照时间间隔进行第一次分割
        4. 对分割后的轨迹段列表进行等时间间隔插值处理
        5. 对插值后的轨迹段进行第二次航行段切割
        6. 划分出对应的训练和测试数据
        // 不进行滤波跳值处理 故而
        mmsi-0 lon-1 lat-2	speed-3	course-4  heading-5	type-6	timestamp-7

        """
        # 0：从最原始的数据源中读取数据 存为dataframe！
        output_file_path = r'./data/ship_FVessel/ais_data.npz'
        # 检查文件是否存在
        if os.path.exists(output_file_path):
            print("文件存在，直接加载数据")
            # 文件存在，直接加载数据
            loaded_data = np.load(output_file_path, allow_pickle=True)
            origin_data = dict(loaded_data)
            # 使用加载的数据进行后续操作
            # for scenario_name, data in data_dict.items():
            #    df = pd.DataFrame(data)  # 将NumPy数组转换为DataFrame
        else:
            print("文件不存在，重新处理保存加载数据")
            FVessel_data_load(output_file_path)
            loaded_data = np.load(output_file_path, allow_pickle=True)
            origin_data = dict(loaded_data)
        # loaded_data可能实际上是一个NpzFile对象，这是一个类似字典的对象。
        # 当你使用dict(loaded_data)将其转换为一个标准的Python字典时，
        # 所有在NpzFile中的项都被转换，包括那些由numpy.savez函数可能自动加入的元数据项。
        # 在这个情况下，'allow_pickle'项可能是一部分元数据。
        if 'allow_pickle' in origin_data:
            del origin_data['allow_pickle']
        new_data = {}
        for video_name,ais_data in origin_data.items():
            # 1. 根据MMSI进行数据分类，排序
            print(video_name+"  processing  ----------")
            new_data[video_name] = FVessel_data_classifer(ais_data)
        # todo 需要添加代码去测试预处理出来的数据的准确性 并合并不同的mmsi为一起 处理成同Marine同样的数据格式 从而复用后续的ship代码
        # 需要再写一个如果数据的列表是空的 则删除数据 ！
        new_data_full = {}
        for video_name,video_dict in new_data.items():
            new_data_single = {}
            for mmsi,list_data in video_dict['ship_data'].items():
                if len(list_data) != 0:
                    new_data_single[mmsi] = video_dict['ship_data'][mmsi]
            new_data_full[video_name] = new_data_single
        # 开始划分训练和测试数据集
        # todo 不处理了 数据本身错误太多 停留段太多 导致最终数据量级几乎太少了 不适合轨迹预测 继续分析 模型本身效果

        print("完成对ship数据的分视频段预处理")
        self.args.seq_length = 15
        self.args.obs_length = 10
        self.args.pred_length = 5
        self.Ship_skip = 10000   # 每次间隔 10000=》10s
        print("完成对原始数据的训练和测试划分！！")



    def data_preprocess_for_transformer(self):
        """
            将数据处理成后续模型transformer的输入形式
            完成原dataPreprocess  工作 适合于无非区分域或则不需要区分域的形式
            dataPreprocess_sne
        """

        pass

    def data_preprocess_for_MLDGtask(self):
        """
            基于data_preprocess_transformer将数据处理成MLDG可以运用的task结构类型
            完成原meta——task工作
        """
        pass

    def get_seq_from_index_balance(self):
        """
            完成get_seq_from_index_balance / get_seq_from_index_balance_meta工作
        """
        pass

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





