import os
import pickle
import random
import time
import numpy as np
import torch
import pandas as pd
import os
import cv2
from copy import deepcopy
from torch import nn

torch.manual_seed(0)


def getLossMask(outputs, node_first, seq_list, using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exsist.
    生成一个掩码，表示当前帧和上一帧中是否都存在数据。该掩码用于计算损失函数时去除缺失数据的贡献，避免缺失数据对损失函数的计算造成影响。
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exsist.
    outputs 是模型的输出，node_first 是形状为 (num_Peds,) 的 Tensor，表示第一帧中存在数据的行人的索引，
    seq_list 是形状为 (seq_length, num_Peds) 的 Tensor，表示每一帧中存在数据的行人的索引。
    函数返回一个形状为 (seq_length, num_Peds) 的 Tensor lossmask 和一个标量 num。其中，lossmask 表示损失掩码，num 表示掩码中元素的数量。
    '''

    if outputs.dim() == 3:
        # [19,257,2]
        seq_length = outputs.shape[0]
    else:
        # 为多次采样而设计的 [20,19,257,2]
        seq_length = outputs.shape[1]

    node_pre = node_first
    lossmask = torch.zeros(seq_length, seq_list.shape[1])

    if using_cuda:
        lossmask = lossmask.cuda()

    # todo ？ For loss mask, only generate for those exist through the whole window
    # 损失的计算只考虑从初始帧开始连续的序列值，空缺帧之后的损失全部不计算
    for framenum in range(seq_length):
        if framenum == 0:
            # 针对于seq-list的第0帧（实际为原始序列的第1帧），node-pre实际为原始序列的第一帧，计算loss，
            # 将该帧与前一帧逐项相乘，若前后帧都存在，则1*1=1，loss-mask的值为1；同样的，其他帧的计算也同理
            lossmask[framenum] = seq_list[framenum] * node_pre
        else:
            # 因为是连续逐帧分析的，那么相应只要有一帧空缺，其后续的将会全部为0，损失计算时不予考虑；
            # 同时需要注意的是序列的第7帧是都存在的，
            lossmask[framenum] = seq_list[framenum] * lossmask[framenum - 1]

    return lossmask, sum(sum(lossmask))


def L2forTest(outputs, targets, obs_length, lossMask):
    '''
    Evaluation.
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[obs_length - 1:, pedi_full]
    error = torch.sum(error_full)
    error_cnt = error_full.numel()
    final_error = torch.sum(error_full[-1])
    final_error_cnt = error_full[-1].numel()

    return error.item(), error_cnt, final_error.item(), final_error_cnt, error_full


def L2forTestS(outputs, targets, obs_length, lossMask, num_samples):
    '''
    Evaluation, stochastic version
    outputs: [20,19,257,2]
    targets:[19,257,2]
    lossmask: [19,257]
    '''
    seq_length = outputs.shape[1]
    #  L2 范数  error (num_samples, seq_length, num_Peds)
    error = torch.norm(outputs - targets, p=2, dim=3)
    # 只提取在整个时间窗口都有数据的行人only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    # 只计算观测序列后面的预测误差总和  (num_samples, pred_length, pedi_full)
    error_full = error[:, obs_length - 1:, pedi_full]
    # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
    error_full_sum = torch.sum(error_full, dim=1)
    # error_full_sum (20,pde-full) error_full_sum_min (pde-full)
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    best_error = []
    # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)
    # error为总的误差 所有行人不同采样中的最佳值累加
    error = torch.sum(error_full_sum_min)
    # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
    error_cnt = error_full.numel() / num_samples
    # 只取终点位置 其为FDE值
    final_error = torch.sum(best_error[-1])
    final_error_cnt = error_full.shape[-1]
    # error: ADE
    # final_error:FDE
    return error.item(), error_cnt, final_error.item(), final_error_cnt

def L2forTest_RMSE_MAE(outputs, targets, obs_length, lossMask, num_samples):
    '''
    Evaluation, stochastic version
    outputs: [20,19,257,2]
    targets:[19,257,2]
    lossmask: [19,257]
    RMSE:
    MAE:
    '''
    seq_length = outputs.shape[1]
    #  L2 范数  error (num_samples, seq_length, num_Peds)
    error = torch.norm(outputs - targets, p=2, dim=3)
    # 只提取在整个时间窗口都有数据的行人only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    # 只计算观测序列后面的预测误差总和  (num_samples, pred_length, pedi_full)
    error_full = error[:, obs_length - 1:, pedi_full]
    # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
    error_full_sum = torch.sum(error_full, dim=1)
    # error_full_sum (20,pde-full) error_full_sum (1,pde-full)
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    best_error = []
    # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)
    # error为总的误差 所有行人不同采样中的最佳值累加
    error = torch.sum(error_full_sum_min)
    # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
    error_cnt = error_full.numel() / num_samples
    # error_rmse 为原mse 开根号；
    error_rmse = torch.sqrt(error)

    # ========MAE
    error_mae = torch.norm(outputs - targets, p=1, dim=3)
    error_mae_full = error_mae[:, obs_length - 1:, pedi_full]
    error_mae_full_sum = torch.sum(error_mae_full, dim=1)
    error_mae_full_sum_min, min_index_mae = torch.min(error_mae_full_sum, dim=0)
    best_error_mae = []
    # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
    for index, value in enumerate(min_index_mae):
        best_error_mae.append(error_mae_full[value, :, index])
    best_error_mae = torch.stack(best_error_mae)
    best_error_mae = best_error_mae.permute(1, 0)
    # error为总的误差 所有行人不同采样中的最佳值累加
    error_mae = torch.sum(error_mae_full_sum_min)

    return error_rmse.item(), error_cnt, error_mae.item(), error_cnt

def L2forTestS_NEWSTAR(prediction, target_pred_traj, num_samples):
    #  L2 范数  error (num_samples, pred_length, num_Peds) inputs (261人 6-43)epoch 0 batch 0 (20 2 138 2)
    error_full = torch.norm(prediction - target_pred_traj, p=2, dim=3)
    # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
    error_full_sum = torch.sum(error_full, dim=1)
    # error_full_sum (20,pde-full) error_full_sum (1,pde-full)
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)
    best_error = []
    # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)
    # error为总的误差 所有行人不同采样中的最佳值累加
    error = torch.sum(error_full_sum_min)
    # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
    error_cnt = error_full.numel() / num_samples
    # 只取终点位置 其为FDE值
    final_error = torch.sum(best_error[-1])
    final_error_cnt = error_full.shape[-1]
    return error.item(),error_cnt,final_error.item(),final_error_cnt

def L2forTest_RMSE_MAE_NEWSTAR(outputs, targets,  num_samples):
    '''
    Evaluation, stochastic version
    outputs: prediction
    targets: target_pred_traj
    outputs: [20,12,257,2]
    targets:[12,257,2]
    RMSE:
    MAE:
    '''
    seq_length = outputs.shape[1]
    #  L2 范数  error (num_samples, pred_length, num_Peds)
    error_full = torch.norm(outputs - targets, p=2, dim=3)
    # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
    error_full_sum = torch.sum(error_full, dim=1)
    # error_full_sum (20,pde-full) error_full_sum (1,pde-full)
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    best_error = []
    # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)
    # error为总的误差 所有行人不同采样中的最佳值累加
    error = torch.sum(error_full_sum_min)
    # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
    error_cnt = error_full.numel() / num_samples
    # error_rmse 为原mse 开根号；
    error_rmse = torch.sqrt(error)

    # ========MAE
    error_mae_full = torch.norm(outputs - targets, p=1, dim=3)
    error_mae_full_sum = torch.sum(error_mae_full, dim=1)
    error_mae_full_sum_min, min_index_mae = torch.min(error_mae_full_sum, dim=0)
    best_error_mae = []
    # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
    for index, value in enumerate(min_index_mae):
        best_error_mae.append(error_mae_full[value, :, index])
    best_error_mae = torch.stack(best_error_mae)
    best_error_mae = best_error_mae.permute(1, 0)
    # error为总的误差 所有行人不同采样中的最佳值累加
    error_mae = torch.sum(error_mae_full_sum_min)

    return error_rmse.item(), error_cnt, error_mae.item(), error_cnt

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te - ts) * 1000, 1), 'ms')
        print()
        return result

    return timed


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.L1Loss()
        loss = criterion(x, y)
        return loss