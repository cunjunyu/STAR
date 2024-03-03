import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import random
sys.path.append(os.getcwd())
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.lines as mlines
from tqdm import tqdm
import pandas as pd
from sklearn import manifold



class NBAConstant:
    """A class for handling constants"""
    # 篮球场的长度为94英尺，宽度为50英尺。此处100-7=93 可能正好对应？ 此处在显示图像时利用了坐标轴的限制的
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 6
    X_MIN = 0
    X_MAX = 100
    Y_MIN = 0
    Y_MAX = 50
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35
    MESSAGE = 'You can rerun the script and choose any event from 0 to '


# 可视化各类图表信息模型
# 不同的数据集不一样 分开多函数处理 NBA和NFL相同类型  SDD和ETH-UCY相同类型


def vis_result(data, args):
    """
    不同的数据集其具体的norm以及尺寸转换可能都不一样 此处需要结合现有模型数据等分析
    需要结合model.inference(data)，首先选取出每个20条轨迹中误差最小的 然后选取出该批次中误差最小的 选择性的画
    选取固定时间窗口内部平均误差最小的进行绘画
    input:
    output:

    """
    dataset = args.dataset


def draw_result_NBA(data, scene, args, mode='pre'):
    print('drawing...')
    traj_num = len(data)
    image_save_path = args.save_dir + 'vis/'
    for idx in range(traj_num):
        plt.clf()
        traj = data[idx]
        traj = traj * 94 / 28  # 重新从m转换回去 需要注意反normalized，反相对于V车的坐标等
        # traj: [15,10,2] -> [10,15,2]
        traj = traj.transpose(1, 0, 2)
        actor_num, length = traj.shape[0], traj.shape[1]
        ax = plt.axes(xlim=(NBAConstant.X_MIN, NBAConstant.X_MAX), ylim=(NBAConstant.Y_MIN, NBAConstant.Y_MAX))
        ax.axis('off')  # 用于关闭坐标轴的显示
        fig = plt.gcf()
        ax.grid(False)
        colorteam1 = 'dodgerblue'
        colorteam2 = 'orangered'
        colorteam1_pre = 'skyblue'
        colorteam2_pre = 'lightsalmon'
        for j in range(actor_num):
            if j < 5:
                color = colorteam1
                color_pre = colorteam1_pre
            elif j < 10:
                color = colorteam2
                color_pre = colorteam2_pre
            for i in range(length):
                points = [(traj[j, i, 0], traj[j, i, 1])]
                (x, y) = zip(*points)  # 将 points 列表中的元组解压缩成两个独立的列表，分别存储在 x 和 y 中。
                # plt.scatter(x, y, color=color,s=20,alpha=0.3+i*((1-0.3)/length))
                if i < 5:
                    plt.scatter(x, y, color=color_pre, s=10, alpha=1)  # 散点的大小为 20，不透明度为 1
                else:
                    plt.scatter(x, y, color=color, s=10, alpha=1)
            for i in range(length - 1):  # 画线
                # 将轨迹中相邻两个点的坐标连接起来，并使用不同的颜色和透明度绘制连线
                points = [(traj[j, i, 0], traj[j, i, 1]), (traj[j, i + 1, 0], traj[j, i + 1, 1])]
                (x, y) = zip(*points)
                # plt.plot(x, y, color=color,alpha=0.3+i*((1-0.3)/length),linewidth=2)
                if i < 4:
                    plt.plot(x, y, color=color_pre, alpha=0.5, linewidth=1)
                else:
                    plt.plot(x, y, color=color, alpha=0.5, linewidth=1)

        court = plt.imread("./data/NBA/nba/court.png")  # 读取图像文件
        plt.imshow(court, zorder=0, extent=[NBAConstant.X_MIN, NBAConstant.X_MAX - NBAConstant.DIFF,
                                            NBAConstant.Y_MAX, NBAConstant.Y_MIN], alpha=0.5)
        if mode == 'pre':
            # plt.savefig('vis/nba/'+str(idx)+'pre.png')
            plt.savefig(image_save_path + str(args.test_set) + str(scene[idx]) + str(idx) + 'pre.png')
        else:
            # plt.savefig('vis/nba/'+str(idx)+'gt.png')
            plt.savefig(image_save_path + str(args.test_set) + str(scene[idx]) + str(idx) + 'gt.png')
        print('finish ' + str(idx) + str(mode) + '.png')
    print('ok')
    return

def find_image_path(scene, image_paths, image_file):
    '''
    根据场景ID在多个图像路径中查找图像文件路径

    :param scene: 场景ID
    :param image_paths: 图像文件所在路径列表
    :param image_file: 图像文件名
    :return: 图像文件路径，如果未找到则返回None
    '''
    for image_path in image_paths:
        im_path = os.path.join(image_path, scene, image_file)
        if os.path.exists(im_path):
            return im_path
    return None

def draw_result_SDD(data,args):
    print('drawing...')
    image_save_path = args.save_dir + 'vis/'
    # 迭代画图
    for index,scene_data in enumerate(data):
        scene_name = scene_data['scene']
        truth_traj = scene_data['truth_traj'].reshape(args.seq_length,2).cpu().numpy() # (20,1,2)->(20,2)
        predict_traj = scene_data['predict_traj'].reshape(args.sample_num,args.pred_length,2).cpu().numpy()  # (20,12,1,2)->(20,12,2)
        # 第一步：先加载选取scene的图像 debug一下
        images_path = ['./data/SDD/train', './data/SDD/test']
        im_path = find_image_path(scene_name, image_paths=images_path, image_file='reference.jpg')
        scene_image = plt.imread(im_path)
        height, width, _ =  scene_image.shape
        # 第二步：画对应的真值轨迹的点
        plt.clf()
        #plt.xlim(0, width)
        #plt.ylim(0, height)
        ax = plt.axes(xlim=(0, width), ylim=(0, height))
        ax.axis('off')  # 用于关闭坐标轴的显示
        fig = plt.gcf()
        ax.grid(False)
        color_truth_past,color_truth_pred, color_pred = 'white','deepskyblue', 'tomato'
        shape_truth_past, shape_truth_pred, shape_color_pred = 's','s','s'

        # 第三步：画对应的20条轨迹的样本值
        for i in range(predict_traj.shape[0]):
            draw_points_and_line(predict_traj[i].reshape(args.pred_length,2),color_pred,shape_color_pred)
        # 画过去真值
        draw_points_and_line(truth_traj[:args.obs_length],color_truth_past,shape_truth_past)
        draw_points_and_line(truth_traj[args.obs_length:], color_truth_pred,shape_truth_pred)
        plt.imshow(scene_image, zorder=0,alpha=1)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(image_save_path + str(args.test_set) + scene_name + str(index) + 'multi_traj.png')
        plt.savefig(image_save_path + str(args.test_set) + scene_name + str(index) + 'multi_traj.svg')
        print('finish ' + str(scene_name) + str(index) + '.png')
    return None

def draw_result_SDD_comparison(data,args):
    print('drawing comparsion with GT,ours,PecNet,GroupNet,AgentFormer,')
    image_save_path = args.save_dir + 'vis_comparison/'
    for index,scene_data in enumerate(data):
        scene_name = scene_data['scene']
        truth_traj = scene_data['truth_traj'].reshape(args.seq_length,2) # (20,1,2)->(20,2)
        predict_traj = scene_data['predict_traj'].reshape(args.sample_num,args.pred_length,2)  # (20,12,1,2)->(20,12,2)
        # 对其进行排序，按误差从小到大 选择4个
        truth_pred_traj = truth_traj[args.obs_length:]
        truth_pred_traj_expanded = truth_pred_traj.unsqueeze(0).expand_as(predict_traj)
        error_full = torch.norm(predict_traj-truth_pred_traj_expanded,p=2,dim=2)
        error_sum = torch.sum(error_full,dim=1)
        # Get the indices of the predictions with the smallest errors
        indices = torch.argsort(error_sum)[:4]
        best_predict_traj = predict_traj[indices]
        # 第一步：先加载选取scene的图像 debug一下
        images_path = ['./data/SDD/train', './data/SDD/test']
        im_path = find_image_path(scene_name, image_paths=images_path, image_file='reference.jpg')
        scene_image = plt.imread(im_path)
        height, width, _ =  scene_image.shape
        plt.clf()
        ax = plt.axes(xlim=(0, width), ylim=(0, height))
        ax.axis('off')  # 用于关闭坐标轴的显示
        fig = plt.gcf()
        ax.grid(False)
        # 过去 白色 真值 红色 best 蓝 绿 橙 粉
        color_truth_past, color_truth_pred, color_pred = 'white', 'tomato', ['#65AE65', '#63A0CB', '#FFA657','#FF927F']
        shape_truth_past, shape_truth_pred, shape_color_pred = 's', 's', 's'
        truth_traj = truth_traj.cpu().numpy()
        best_predict_traj = best_predict_traj.cpu().numpy()
        for i in range(best_predict_traj.shape[0]):
            draw_points_and_line(best_predict_traj[i].reshape(args.pred_length,2),color_pred[i],shape_color_pred)
        # 画过去真值
        draw_points_and_line(truth_traj[:args.obs_length], color_truth_past, shape_truth_past)
        draw_points_and_line(truth_traj[args.obs_length:], color_truth_pred, shape_truth_pred)
        plt.imshow(scene_image, zorder=0,alpha=1)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(image_save_path + str(args.test_set) + scene_name + str(index) + 'comparison_traj.png')
        plt.savefig(image_save_path + str(args.test_set) + scene_name + str(index) + 'comparison_traj.svg')
        print('finish ' + str(scene_name) + str(index) + '.png')
    return None

def SDD_traj_vis():
    """
    作用为相应的可视化每个场景中的所有轨迹数据 只取行人数据

    """
    # 第一步 简化数据 剔除非行人 剔除部分不需要的列
    DATA_PATH = '../data/SDD/sdd_full_data.pkl'
    sdd_data = pd.read_pickle(DATA_PATH)
    # 同时尝试将所有的加上车辆和其他的轨迹数据也画一遍
    # sdd_ped_data = sdd_data[sdd_data['label']=='Pedestrian']
    sdd_ped_data = sdd_data
    sdd_ped_data = sdd_ped_data.drop(columns=['scene', 'trackId'])
    grouped_scene = sdd_ped_data.groupby('sceneId')
    for scene_name, scene_data in grouped_scene:
        # 依据每个不同的metai筛选出数据
        # 按照metaid分组并按时间排序
        print('process data in ' + str(scene_name))
        sorted_scene_data = scene_data.groupby('metaId').apply(lambda x: x.sort_values('frame'))
        # 提取排序后的时间序列数据
        time_series_data = sorted_scene_data[['frame', 'x', 'y', 'metaId']]
        # 重置索引以明确'metaId'列
        time_series_data.reset_index(drop=True, inplace=True)
        # 创建一个图形对象
        fig, ax = plt.subplots()
        # 需要统一相应的画图坐标轴大小 x:-1500-》+1500     y:-1500-》+1500
        # 统一的x轴和y轴范围
        plt.xlim(-1500, 1500)
        plt.ylim(-1500, 1500)
        for metaId_name, metaId_data in time_series_data.groupby('metaId'):
            # 将起点设置为(0, 0)
            metaId_data['x'] = metaId_data['x'].transform(lambda x: x - x.iloc[0])
            metaId_data['y'] = metaId_data['y'].transform(lambda x: x - x.iloc[0])
            ax.plot(metaId_data['x'], metaId_data['y'], label=f'MetaID {metaId_name}')
        # 添加图例和标题
        # ax.legend()
        ax.set_title(scene_name)
        # 显示图形
        print('finsih draw ' + str(scene_name))
        # 图像保存为svg格式 从而方便后续修改
        # image_save_path = '../data/SDD/vis_traj/'+str(scene_name)+'.png'
        # plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        image_save_path = '../data/SDD/vis_traj/' + str(scene_name) + '.svg'
        plt.savefig(image_save_path, format='svg', bbox_inches='tight')

    # 关闭最后一个图形对象
    if fig is not None:
        plt.close(fig)
    print('finish drawing all picture !!')

def draw_points_and_line(data, color,shape):
    # 输入的data为（length，2）
    length = data.shape[0]
    for i in range(length):
        # points = [data[i, 0], data[i, 1]]
        #(x, y) = zip(*points)  # 将 points 列表中的元组解压缩成两个独立的列表，分别存储在 x 和 y 中。
        plt.scatter(data[i, 0], data[i, 1], color=color,marker=shape, s=10, alpha=1,edgecolors=color, facecolors='')  # 散点的大小为 10，不透明度为 1
    for i in range(length - 1):  # 画线
        # 将轨迹中相邻两个点的坐标连接起来，并使用不同的颜色和透明度绘制连线
        # points = [(data[i, 0], data[i, 1]), (data[i + 1, 0], data[i + 1, 1])]
        # (x, y) = zip(*points)
        plt.plot([data[i, 0],data[i + 1, 0]],[data[i, 1],data[i + 1, 1]], color=color, alpha=0.5, linewidth=1)

# T-SNE可视化代码
"""
T-SNE可视化 针对于高维数据的降维与可视化 
将多个域的数据进行混合？
首先对eth-ucy中训练时刻已有的多个域数据进行可视化，可视化的点在相应的encoder结尾。
==》具体分析：不好的情况：测试域的数据与4个训练域都相差很远=》MVDG后 测试域数据与4个训练域相比较近  
    故而 在测试某个数据域的时候 对应的同样处理其他四个数据集 并分开保存 
    在测试的过程中保存下各自的past-feature 继而存下来后续画图  
    ==》基于ETH-UCY
    第一步：添加test-sne参数 若正确则处理完整的数据集为test-sne
    第二步：在相应的test-epoch与inference中进行逐数据集处理 并保存对应的
    第三步：
"""
def TSNE(feat,perplexity):
    # input：feat : [num, dim]  output ： [num,2]
    # n_components, # 降维后嵌入空间的维度，如2或3
    # init,         # 嵌入的初始化，可选'pca'或'random'，默认pca，pca效果会更好
    # random_state, # 伪随机数发生器种子控制
    # perplexity（默认值：30）：perplexity 与其他流形学习算法中使用的最近邻的数量有关。考虑选择 5 到 50 之间的值。
    # n_iter（默认值：1000）：优化的最大迭代次数。应至少为 250

    tsne = manifold.TSNE(n_components=2, init='pca',perplexity=perplexity,random_state=12)
    x_ts = tsne.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    # t-SNE 归一化
    #x_min, x_max = x_ts.min(0), x_ts.max(0)
    #x_final = (x_ts - x_min) / (x_max - x_min)
    x_final = x_ts
    return x_final

def VISUAL_TSNE(data_dict,args):
    # 设置散点形状
    maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    perplexity = 20
    # 设置散点颜色
    colors_sum ={'warm':[ '#FF7F50','#FFA500','#FF4500','#FF6347','#FF8C00'],# 不好看
                 'cool':['#00BFFF','#87CEEB','#4169E1','#0000FF','#00008B'],
                 'earthy':['#8B4513','#CD853F','#DAA520','#A0522D','#8B0000'],
                 'bright':['#FF0000','#FFD700','#00FF00','#00FFFF','#FF00FF'],
                 'pastel':['#FF9AA2','#FFB7B2','#FFDAC1','#E2F0CB','#B5EAD7'],# 还行
                 'pink':['#D86967','#58539F','#BBBBD6','#EEBABB'],
                 'try':['deepskyblue','springgreen','tomato','gold','orange','pink'],
                 'try2':['lightskyblue','mediumspringgreen','salmon','orange'],
                 'try3':['#feb64d','tomato','springgreen','deepskyblue','gold'],
                 'good':['#32d3eb','#5bc49f','#feb64d' ,'#ff7c7c','#9287e7','#60acfc']
                 }
    # colors = ['red', 'yellow', 'green','blue', 'black']
    colors = colors_sum['try']
    # 设置字体格式
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }
    # 拼接对应的array在一起 并添加单独的label标签值
    full_scene_data = np.concatenate(list(data_dict.values()),axis=0)
    scene_list = list(data_dict.keys())
    full_scene_label = []
    for scene,scene_data in data_dict.items():
        scene_label = [scene for _ in range(len(scene_data))]
        full_scene_label.extend(scene_label)
    # full_scene_label =  np.concatenate([np.full((data_dict[key].shape[0],), i) for i, key in enumerate(data_dict)], axis=0)
    full_scene_label = np.array(full_scene_label).reshape((-1,1))
    print('begin process data to SNE')
    full_scene_data_sne = TSNE(full_scene_data,perplexity)
    data = np.hstack((full_scene_data_sne,full_scene_label))
    pd_data = pd.DataFrame({'x': data[:, 0].flatten().astype(float), 'y': data[:, 1].flatten().astype(float), 'label': data[:, 2].flatten()})
    # 在绘制散点图之前添加以下代码进行降采样
    # 对zara02降采样
    # pd_data = pd_data[pd_data['label'] != 'zara02']
    zara02_data = pd_data[pd_data['label'] == 'zara02']
    sampled_zara02_data = zara02_data.sample(n=2000, random_state=42)  # 从 "zara02" 数据中随机选择 2500 个样本
    other_data_zara02 = pd_data[pd_data['label'] != 'zara02']
    pd_data_sampled_zara02 = pd.concat([sampled_zara02_data, other_data_zara02])  # 将降采样后的数据与其他数据合并
    pd_data = pd_data_sampled_zara02
    # 对univ降采样
    # pd_data = pd_data[pd_data['label']!='univ']
    univ_data = pd_data[pd_data['label']=='univ']
    sampled_univ_data = univ_data.sample(n=1000,random_state=42)
    other_data_univ = pd_data[pd_data['label']!='univ']
    pd_data_sampled_univ = pd.concat([sampled_univ_data,other_data_univ])
    pd_data = pd_data_sampled_univ

    scene_list_new = pd_data['label'].unique()
    print(pd_data.shape)
    # 社遏制坐标轴大小
    plt.clf()
    x_min, x_max = pd_data['x'].min(), pd_data['x'].max()
    y_min, y_max = pd_data['y'].min(), pd_data['y'].max()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for index,scene in enumerate(scene_list_new):
        print('drawing '+str(scene))
        X = pd_data.loc[pd_data['label'] == scene]['x']
        Y = pd_data.loc[pd_data['label'] == scene]['y']
        plt.scatter(X, Y, cmap='brg', s=3, marker='o', c=colors[index], edgecolors=colors[index], alpha=0.65)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
    legend_labels = [scene_list_new[i] for i in range(len(scene_list_new))]
    plt.legend(legend_labels, loc='best', prop=font1)
    name = str(args.test_set)+'_'+str(args.stage)+'_epoch_'+str(args.load_model)
    plt.title(name, fontsize=10, fontweight='normal', pad=20)
    image_save_path = args.save_base_dir + str(args.dataset)+'/' + 'vis_sne/'
    plt.savefig(image_save_path + str(args.test_set) + str(perplexity)+ str(args.stage) + str(args.load_model) + 'sne.png')
    plt.savefig(image_save_path + str(args.test_set) + str(perplexity)+ str(args.stage) + str(args.load_model) + 'sne.svg')
    print('finish drwaing ')


# 轨迹可视化
"""
针对于不同数据集下不同的轨迹可视化方法
1：多模态可视化方法 每张图上包含一条过去轨迹以及相应的多模态未来轨迹
2：因为数据在处理过程中被叠加 失去了对应的域信息，故而此处相应的增加一个列表去存储 并与batch的inputs随行
3：先以SDD数据处理进行进行尝试

"""

"""
def test_new_visualNBA_epoch(self):
    self.dataloader.reset_batch_pointer(set='test')
    error_epoch, final_error_epoch = 0, 0,
    error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
    visual_batch_inputs = []
    visual_batch_outputs = []
    visual_batch_error = []
    for batch in tqdm(range(self.dataloader.testbatchnums)):
        # 第一步：获取对应的预测结果 预测结果取的是20次预测中每个独立个体最好的一次 然后进行保存
        # 与train相同的batch处理步骤
        inputs, batch_id = self.dataloader.get_test_batch(batch)
        inputs = tuple([torch.Tensor(i) for i in inputs])
        if self.args.using_cuda:
            inputs = tuple([i.cuda() for i in inputs])
        with torch.no_grad():
            prediction, target_pred_traj = self.net.inference(inputs)
            # prediction (sample_num, agent_num, self.pred_length, 2) -> (sample_num, self.pred_length,agent_num, 2)
            # target_pred_traj  (self.pred_length,agent_num, 2)
            prediction = prediction.permute(0, 2, 1, 3)
        # 过去轨迹inputs[0] nodes_abs [15,520,2] 真实的轨迹[10,520,2] 完全对应的 预测的轨迹 prediction [20,10,520,2] 此处的520 是因为设置512个行人数据
        # 所有的行人被统一拼接到对应的20帧内 则后续需要绘画结果时 无法依照时间和地点去取出对应的值
        # 设计格式 [meta-id,start-frame,end-frame,sceneid]得添加一个数组 标注每个inputs相应的name和id；从而方便后期依照对应的id和name从统一的保存的预测结果中索引对应的值
        # 针对于NBA数据集 可以进行相应的简化 因为其每次都是10个人 分析清楚如何拆分的 即可将原始数据进行拆分
        error_full = torch.norm(prediction - target_pred_traj, p=2, dim=3)  # 计算xy对应的误差 由【20,10,520,2】转变成error_full【20,10,520】
        error_full_sum = torch.sum(error_full, dim=1)  # 汇总这10s的总误差 【20，10,520】-》error_full_sum 【20.520】
        error_full_sum_min, min_index = torch.min(error_full_sum, dim=0) # 从20个选出最好的一个 【20,520】 -> 【520】
        for index, value in enumerate(min_index):
            best_error.append(error_full[value, :, index])
            prediction_mindex = prediction[value,:,index,:]
        best_error = torch.stack(best_error)
        best_error = best_error.permute(1, 0)
        # error为总的误差 所有行人不同采样中的最佳值累加
        error = torch.sum(error_full_sum_min)
        # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
        error_cnt = error_full.numel() / self.args.sample_num
        # 只取终点位置 其为FDE值
        final_error = torch.sum(best_error[-1])
        final_error_cnt = error_full.shape[-1]
        error_epoch += error.item()
        error_cnt_epoch += error_cnt
        final_error_epoch += final_error.item()
        final_error_cnt_epoch += final_error_cnt

        # 依照10人一组 将这个520拆分开来并汇总误差 选取最好的前3个和最差的1个 两轮选取 第一轮在每个batch内部 第二轮在整个数据集选取
        error_full_summin_reshaped = error_full_sum_min.reshape(52,10)
        group_sums = torch.sum(error_full_summin_reshaped,dim=1)
        top_2_indices = torch.topk(group_sums,k=3,largest=False).indices
        worst_index = torch.argmax(group_sums)
        # 从对应的原始预测数据中取出 历史值和未来预测值
        for index,value in enumerate(top_2_indices):
            visual_batch_inputs.append(inputs[0][:,value*10:(value+1)*10,:])
            # 拼接历史的origin与预测的prediction 成为完整数据
            full_prediction_withinputs = torch.cat((inputs[0][0:5,value*10:(value+1)*10,:],prediction_mindex[:,value*10:(value+1)*10,:]),dim=0)
            visual_batch_outputs.append(full_prediction_withinputs)
            visual_batch_error.append(group_sums[value])
    #  依据所有的实验batch数据 在进行一次排序 选取相应的最小的5个error数据
    min_indices_all = [index for index, value in sorted(enumerate(visual_batch_error), key=lambda x: x[1])[:5]]
    gt_data,pre_data,scene = [],[],[]
    for index,value in enumerate(min_indices_all):
        gt_data.append(visual_batch_inputs[value])
        pre_data.append(visual_batch_outputs[value])
        scene.append('nba')
    self.visual_dict = {'gt_data':gt_data,'pre_data':pre_data,'scene':scene}
    return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
"""
# 假设您的变量为：new_scene_frame（DataFrame，115x2）、error（numpy数组，长度为115）、prediction（numpy数组，形状为115x12x2）


