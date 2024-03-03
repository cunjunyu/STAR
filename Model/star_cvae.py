import copy

import numpy as np
import torch
import torch.nn as nn
from Model.CVAE_utils import Normal,MLP2,MLP
from torch.distributions.normal import Normal as Normal_official
from Model.star import TransformerModel,TransformerEncoder,TransformerEncoderLayer

torch.manual_seed(0)


class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''

    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)

        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_size=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim, past_len * 2, hidden_size=(512, 256))

        self.relu = nn.ReLU()

        # kaiming initialization
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)

    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, D

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        # 和公式保持一致，其block1因为x-hat为0，故而x-=x_true；block2则为x_true-x_hat（上一个block预测输出）
        x_ = x_true - x_hat
        # (128*11,5,2) -> (128*11,2,5)
        x_ = torch.transpose(x_, 1, 2)
        # 首先将x-（过去轨迹数据）经过CNN层（将xy2维编码成32维）； (128*11,2,5) ->  (128*11,32,5)
        past_embed = self.relu(self.conv_past(x_))
        #  (128*11,32,5) ->  (128*11,5,32)
        past_embed = torch.transpose(past_embed, 1, 2)
        # 而后输入GRU层中，利用时序信息的处理，5个时间点一步步转为为最终的输出，即又（128*11,5,32）--（1,128*11,96）
        _, state_past = self.encoder_past(past_embed)
        # （1,128*11,96）-> （128*11,96）
        state_past = state_past.squeeze(0)
        # f为拼接z和经过GroupNet编码的past-feature的拼接诶，此处input-feat再度将由原始past-traj经过CNN和GRU提出的state-past拼接起来
        input_feat = torch.cat((f, state_past), dim=1)
        # decoder_x,decoder_y皆为带有两层隐藏层的MLP结构
        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)

        return x_hat_after, y_hat


class Decoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # todo 值得应该是past的维度
        self.model_dim = 32
        self.decode_way = 'RES'
        self.num_decompose = 2
        input_dim = self.model_dim + self.args.zdim
        self.past_length = self.args.obs_length
        self.future_length = self.args.pred_length
        self.decompose = nn.ModuleList(
            [DecomposeBlock(self.past_length, self.future_length, input_dim) for _ in range(self.num_decompose)])

    def forward(self, past_feature, z, past_traj, sample_num, mode='train'):
        # past_feature ( batch_size, hidden_dim) z  past_traj ((259,8，2)未归一化的？初步是的，但是past-feature是归一化后产生的)
        agent_num = past_traj.shape[0]
        # past_traj ,(259,8,2)->(259*sample_num,8,2)
        # 修改cur_location使得其与pred匹配
        cur_location = past_traj[:, -1, :]
        cur_location = cur_location.view(agent_num, 1, cur_location.shape[-1])
        cur_location = cur_location.repeat_interleave(self.future_length, dim=1)
        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        past_traj_repeat = past_traj.repeat_interleave(sample_num, dim=0)
        past_feature = past_feature.view(-1, sample_num, past_feature.shape[-1])
        z_in = z.view(-1, sample_num, z.shape[-1])
        # 拼接z和past-features (bath,sample_num,z_in+past_feature)
        hidden = torch.cat((past_feature, z_in), dim=-1)
        # 合并agent-num*sample——num方便计算 （128*11,256+32）
        hidden = hidden.view(agent_num * sample_num, -1)
        x_true = past_traj_repeat.clone()
        x_hat = torch.zeros_like(x_true)
        # 此处对应的batch-size即STAR预处理完的行人数量
        batch_size = x_true.size(0)
        prediction = torch.zeros((batch_size, self.future_length, 2)).cuda()
        reconstruction = torch.zeros((batch_size, self.past_length, 2)).cuda()
        # 上述一系列主要为重复past-traj共sample-num次，重构past-feature和z并将他们拼接作为图中的V-out即hidden；相应的设置好past-true，predict等格式
        # 循环对子模块进行迭代，分别得到预测值和y-hat和重构值x-hat，这些预测值和重构值被累加到prediction和reconstruction张量中，用于最终输出
        # 对应于解码过程的Fblock1和Fblock2，以及两个解码块得到结果后进行累加。
        for i in range(self.num_decompose):
            x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)
            prediction += y_hat
            reconstruction += x_hat
        # 将prediction重新构造为out-seq，并加上当前位置cur-location-repeat，得到最终的输出张量
        norm_seq = prediction.view(agent_num * sample_num, self.future_length, 2)
        recover_pre_seq = reconstruction.view(agent_num * sample_num, self.past_length, 2)
        out_seq = norm_seq + cur_location_repeat
        # todo need to test
        if mode == 'inference':
            out_seq = out_seq.view(-1, sample_num, *out_seq.shape[1:])
            # (agent_num, sample_num, self.pred_length, 2)
        return out_seq, recover_pre_seq


class STEncoder(torch.nn.Module):
    """
    输入的是past相关的数据，
    """

    def __init__(self, args, stage, dropout_prob=0):
        super(STEncoder, self).__init__()
        self.embedding_size = 32
        self.dropout_prob = dropout_prob
        self.args = args
        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward network ModelStrategy in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value

        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.temporal_encoder_1 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        self.temporal_encoder_2 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)
        # Linear layer to output and fusion
        self.fusion_layer = nn.Linear(64, 32)
        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in_1 = nn.Dropout(self.dropout_prob)
        self.dropout_in_2 = nn.Dropout(self.dropout_prob)
        # self.time_embedding = self.args.time_embedding
        if stage == 'past':
            self.encoder_full_layer = nn.Linear(self.args.obs_length, 1)
        elif stage == 'future':
            self.encoder_full_layer = nn.Linear(self.args.pred_length, 1)

    def forward(self, inputs):
        # nodes_current/abs (8,num_ped,2),nei_list(8，num_ped，num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]
        num_ped = nodes_current.shape[1]
        # 第一层frame
        temporal_input_embedded = self.dropout_in_1(self.relu(self.input_embedding_layer_temporal(nodes_current)))
        temporal_input_embedded_1 = self.temporal_encoder_1(temporal_input_embedded)
        spatial_input_embedded = self.dropout_in_2(self.relu(self.input_embedding_layer_spatial(nodes_abs_position)))
        spatial_embedded_2 = torch.zeros(length, num_ped, self.embedding_size).cuda()
        # 第一层encoder1
        for frame in range(length):
            # 在第一轮拼接分别经过空间和时间编码的特征帧
            spatial_input_embedded_1 = self.spatial_encoder_1(spatial_input_embedded[frame].unsqueeze(1),
                                                              nei_list[frame])
            # (num_ped,1,32)->(num-ped,32)
            spatial_input_embedded_frame = spatial_input_embedded_1.permute(1, 0, 2)[-1]
            temporal_input_embedded_frame = temporal_input_embedded_1[frame]
            fusion_feat = torch.cat((temporal_input_embedded_frame, spatial_input_embedded_frame), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)
            # 第一轮结束后，第二轮的空间encoder
            spatial_input_embedded_2 = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list[frame])
            spatial_input_embedded_2 = spatial_input_embedded_2.permute(1, 0, 2)[-1]
            spatial_embedded_2[frame] = spatial_input_embedded_2
        # 第二层时间encoder2 -> temporal_input_embedded_2 (8,num_ped,32)
        temporal_input_embedded_2 = self.temporal_encoder_2(spatial_embedded_2)
        # todo 整合时序关系 !! 只用最后一帧或者全部帧经过mlp并无区别
        spatial_temporal_embedded = temporal_input_embedded_2.permute(1, 2, 0)  # -> (num_ped,32,8)
        spatial_temporal_embedded = self.encoder_full_layer(spatial_temporal_embedded)
        spatial_temporal_embedded = spatial_temporal_embedded.view(num_ped, self.embedding_size)
        """
        if self.time_embedding:
            spatial_temporal_embedded = self.encoder_full_layer(spatial_temporal_embedded)
            spatial_temporal_embedded = spatial_temporal_embedded.view(num_ped,self.embedding_size)
        else:
            spatial_temporal_embedded = spatial_temporal_embedded[:,:,-1].view(num_ped,self.embedding_size)
        """
        return spatial_temporal_embedded


class STAR_CVAE(torch.nn.Module):
    """
    输入的是完整的数据，而后基于完整过去数据，拆分成过去和未来，以及Decoder
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_size = 32
        # modeule structure
        self.past_encoder = STEncoder(args, stage='past')
        self.future_encoder = STEncoder(args, stage='future')
        self.out_mlp = MLP2(input_dim=64, hidden_dims=[32], activation='relu')
        # todo 此处维度需要注意 可能有问题 后期调节超参数
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        # todo 注意此处结合分析pz-layer 即学习先验的必要性
        # self.pz_layer = nn.Linear(self.embedding_size, 2*self.args.zdim)
        self.decoder = Decoder(args)

    def get_node_index(self, seq_list):
        """

        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):

            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices

    def update_batch_pednum(self, batch_pednum, ped_list):
        """

        :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
        :type list
        :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
        :type tensor
        :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
        :rtype: list
        """
        updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
        updated_batch_pednum = copy.deepcopy(batch_pednum)

        cumsum = np.cumsum(updated_batch_pednum_)
        new_ped = copy.deepcopy(ped_list).cpu().numpy()

        for idx, num in enumerate(cumsum):
            num = int(num)
            if idx == 0:
                updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
            else:
                updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

        return updated_batch_pednum

    def mean_normalize_abs_input(self, node_abs, st_ed):
        """
        目标：是为了进行空间位置的归一化 即将属于同一个场景下的不同agent的轨迹数据归一化到同一个原点 这个原点的计算由不同agent轨迹数据的全部值共同决定
        :param node_abs: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: node_abs: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        node_abs = node_abs.permute(1, 0, 2)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)

    def get_st_ed(self, batch_num):
        """

        :param batch_num: contains number of pedestrians in different scenes for a batch
        :type batch_num: list
        :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
        :rtype: list
        """
        cumsum = torch.cumsum(batch_num, dim=0)
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

        st_ed.insert(0, (0, int(cumsum[0])))

        return st_ed

    def calculate_loss_diverse(self, pred, target):
        # 用于计算损失的方法。它接收预测值、目标值和批次大小，并计算每个预测值与目标值之间的欧几里得距离。
        # 然后，对于每个目标值，选择最小距离并取平均值作为损失。最终返回损失值。
        # target: (batch,pred,2) ->   (batch,1,pred,2)   pred: (batch,sample_num,pred,2) diff ；(batch,sample_num,pred,2)
        diff = target.unsqueeze(1) - pred
        # avg_dist （batch，sample_num）计算各目标对应距离平方和
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        # 提取每个实例各目标最小距离
        loss = avg_dist.min(dim=1)[0]
        loss = loss.mean()
        return loss

    def calculate_loss_kl(self, qz_distribution, pz_distribution, batch_ped, min_clip):
        loss = qz_distribution.kl(pz_distribution).sum()
        loss /= (batch_ped)
        loss_clamp = loss.clamp_min_(min_clip)
        return loss_clamp

    def calculate_loss_pred(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss

    def calculate_loss_recover(self, pred, target, batch_size):
        loss = (target - pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss

    def forward(self, inputs, stage, mean_list=[], var_list=[], ifmixup=False):
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 注意 此处针对的是不同帧下行人数量不一的问题，传统的解决思路是直接提取从头到尾到存在的行人，而star方法则是逐帧中提取从头到尾到存在的行人，【应该是将空的行人进行补零了】
        # 最后的loss计算时也是只考虑从头到尾的行人，这个只是相当于利用了更多行人的信息；但在此处，我们采用传统思路预处理，因为我们是针对完全形态分析
        # node-idx筛选出从起始到当前帧都存在的ped
        node_index = self.get_node_index(seq_list)
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]
        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        # todo 在batch41出现batch_pednum = 0 的情况 ！ 应该利用断言去写！
        if batch_pednum.cpu().detach().numpy().shape[0] - updated_batch_pednum.cpu().detach().numpy().shape[0] > 0:
            print(
                'batch_pednum:' + str(batch_pednum.cpu().detach().numpy().shape) + '/' + 'updated_batch_pednum:' + str(
                    batch_pednum.cpu().detach().numpy().shape))

        st_ed = self.get_st_ed(updated_batch_pednum)
        # todo 提取新的轨迹数据 提取的是未归一化的
        nodes_current = nodes_abs[:, node_index, :]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_abs[:, node_index], st_ed)
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:obs_length]
        future_traj = nodes_current[obs_length:], nodes_abs_position[obs_length:], nei_list[obs_length:]
        # (num_ped,hidden_feature32)
        past_feature = self.past_encoder(past_traj)
        future_feature = self.future_encoder(future_traj)
        #  添加HIN的话 基于上述内容进行
        # ---------------------CVAE-------------------------------
        ### q dist ###
        """
        根据超参数self.args.ztype的设置，选择创建哪种分布类型的后验分布q(z|x,y)。如果ztype为'gaussian'，
        则使用Normal（高斯分布）类创建一个正态分布，并以qz_param作为分布的参数;从后验分布q(z | x,y)中抽样得到qz_sampled，将其用于计算KL散度（KL divergence）损失项。
        """
        # (batch_size,hidden_dim*2) 64
        h = torch.cat((past_feature, future_feature), dim=1)
        # (batch_size,32) 64->32
        h = self.out_mlp(h)
        # 在变分自编码器中，潜变量z的均值和方差是通过编码器网络输出的，并且需要满足一定的分布假设，例如高斯分布。
        # 因此，该线性层的作用是将 MLP 的输出映射到满足分布假设的潜变量均值和方差，从而使得潜变量 z 可以被正确地解码和重构。
        qz_param = self.qz_layer(h)
        if self.args.ztype == 'gaussian':
            qz_distribution = Normal(params=qz_param)
        else:
            ValueError('Unknown hidden distribution!')
        # qz_sampled （num_ped,self.zdim16）
        qz_sampled = qz_distribution.rsample()
        ### todo p dist ###
        """
        初步调节分析 不需要先验分布
        根据超参数self.args.learn_prior的设置，选择创建哪种分布类型的先验分布p(z)。如果learn_prior为True，
        则使用线性层对象self.pz_layer生成一个包含均值和标准差参数的向量pz_param，并以此创建一个Normal分布pz_distribution。
        如果learn_prior为False，则以0为均值、方差为1的标准正态分布作为先验分布p(z)

        if self.args.learn_prior:
            pz_param = self.pz_layer(past_feature)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=pz_param)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            if self.args.ztype == 'gaussian':
                # todo 分析出 mu logvar sigma 用法
                pz_distribution = Normal(mu=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_feature.device),
                                         logvar=torch.zeros(past_feature.shape[0], self.args.zdim).to(
                                             past_feature.device))
            else:
                ValueError('Unknown hidden distribution!')
        """
        if self.args.ztype == 'gaussian':
            # todo 分析出 mu logvar sigma 用法
            pz_distribution = Normal(mu=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_feature.device),
                                     logvar=torch.zeros(past_feature.shape[0], self.args.zdim).to(
                                         past_feature.device))
        else:
            ValueError('Unknown hidden distribution!')

        ### use q ###
        # z = qz_sampled 基于解码器，输入过去轨迹past_traj()和特征past_feature()，采样的z，agent_num
        # todo 即node-past此处没有经历过依场景空间归一化，而past-traj输入有归一化后的数据 故而需要分析此处的node-past选取的合理性？？
        node_past = nodes_current[:obs_length].transpose(0, 1)  # (num-ped,8,2)
        node_future = nodes_current[obs_length:].transpose(0, 1)  # (num-ped,12,2)
        # pred:(num-ped,pred,2)  recover (num_ped,obs,2)
        # todo mixup
        '''
        mixup 在此处做 Z是特殊噪声项的引入，而且为特殊设计，通用性不强；故而选择在Past-feature处做；
        在元训练阶段，记录每批的均值和方差，进行存储；而后再测试阶段加入即可
        '''
        mean, var = 0, 0
        if ifmixup:
            if stage == 'support':
                mean = past_feature.mean(dim=0)
                var = past_feature.var(dim=0)
                pred_traj, recover_traj = self.decoder(past_feature, qz_sampled, node_past, sample_num=1)
                assert pred_traj.shape[0] == node_future.shape[0] == recover_traj.shape[0] == node_past.shape[0]
                batch_size = pred_traj.shape[0]
                loss_pred = self.calculate_loss_pred(pred_traj, node_future, batch_size)


            elif stage == 'query':
                meta_train_domain = len(mean_list)
                loss_pred_list = []
                for i in range(meta_train_domain):
                    Distri = Normal_official(mean_list[i], var_list[i])
                    sample = Distri.sample([past_feature.size(0), ])
                    # todo 注意lam取值
                    lam = np.random.beta(1., 1.)
                    final_feature = past_feature * (1 - lam) + lam * sample
                    pred_traj, recover_traj = self.decoder(final_feature, qz_sampled, node_past, sample_num=1)
                    batch_size = pred_traj.shape[0]
                    loss_pred = self.calculate_loss_pred(pred_traj, node_future, batch_size)
                    loss_pred_list.append(loss_pred)
                loss_pred = torch.mean(torch.stack(loss_pred_list))

            loss_recover = self.calculate_loss_recover(recover_traj, node_past, batch_size)
            batch_pednum = past_feature.shape[0]
            loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_pednum, self.args.min_clip)

            # Dirichet（a）分布 后续考虑 此处先设定为平均相加
            """
            RG = np.random.default_rng()
            mixuplist =  [1/meta_train_domain for _ in range(meta_train_domain)]
            # past-feature (209,32)->mixup-ratio(209,32,4)==>(4,209,32)
            mixup_ratio = RG.dirichlet(mixuplist,past_feature.shape)
            mixup_ratio = torch.from_numpy(mixup_ratio).cuda().permute(2,0,1).float()
            mixup_feature = torch.stack(sample_list)
            mixup_domain_feature = torch.sum(mixup_feature*mixup_ratio,dim=0)
            lam = self.args.lam
            final_feature = past_feature*(1-lam)+lam*mixup_domain_feature
            past_feature = final_feature
            """
        else:
            pred_traj, recover_traj = self.decoder(past_feature, qz_sampled, node_past, sample_num=1)
            assert pred_traj.shape[0] == node_future.shape[0] == recover_traj.shape[0] == node_past.shape[0]
            batch_size = pred_traj.shape[0]
            loss_recover = self.calculate_loss_recover(recover_traj, node_past, batch_size)
            loss_pred = self.calculate_loss_pred(pred_traj, node_future, batch_size)
            batch_pednum = past_feature.shape[0]
            loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_pednum, self.args.min_clip)

        ### p dist for best 20 loss ###
        """
        主要目的在于计算loss-divers -- 由Social-GAN提出来的
        根据均值和标准差参数p_z_params（如果self.args.learn_prior为True，则使用线性层对象self.pz_layer生成），创建先验分布p(z)，用于计算ELBO损失。
        区别在于，这里对每个样本采样sample_num次，以便在计算ELBO损失时可以更精确地估计期望值。
        将past_feature张量重复sample_num次，以便将batch_size和agent_num两个维度扩展为(batch_size * agent_num * sample_num)。
        """
        sample_num = 20
        """
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device),
                    logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device))
            else:
                ValueError('Unknown hidden distribution!')
        """
        # todo pz-layer
        past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
        if self.args.ztype == 'gaussian':
            pz_distribution = Normal(
                mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device),
                logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device))
        else:
            ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        # diverse_pred_traj  (agent_num, sample_num, self.past_length, 2)
        diverse_pred_traj, _ = self.decoder(past_feature_repeat, pz_sampled, node_past, sample_num=20, mode='inference')

        loss_diverse = self.calculate_loss_diverse(diverse_pred_traj, node_future)
        total_loss = loss_pred + loss_recover + loss_kl + loss_diverse

        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item(), mean, var

    def inference(self, inputs):
        # 这是一个模型推理方法的实现。该方法接收包含过去轨迹的数据，并根据学习到的先验和给定的过去轨迹输出多样化的预测轨迹。
        # 该方法使用过去编码器计算过去特征，使用pz_layer从先验分布中采样，并使用解码器生成多样化的预测轨迹。最终输出的结果是多样性预测轨迹。
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 注意 此处针对的是不同帧下行人数量不一的问题，传统的解决思路是直接提取从头到尾到存在的行人，而star方法则是逐帧中提取从头到尾到存在的行人，
        # 最后的loss计算时也是只考虑从头到尾的行人，这个只是相当于利用了更多行人的信息；但在此处，我们采用传统思路预处理，因为我们是针对完全形态分析
        # node-idx筛选出从起始到当前帧都存在的ped
        node_index = self.get_node_index(seq_list)
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]
        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index, :]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_abs[:, node_index], st_ed)
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:obs_length]
        past_feature = self.past_encoder(past_traj)
        target_pred_traj = nodes_current[obs_length:]
        # 上述代码计算潜在空间中的特征向量past_feature，相同于forward，只是在测试推理过程中不需要未来数据
        sample_num = self.args.sample_num
        """
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(
                    mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device),
                    logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device))
            else:
                ValueError('Unknown hidden distribution!')
        """
        # todo pz-layer有问题 这里的inference与train阶段的pz-layer完全没有联系 则完全未学习 是一个随机的噪声？？
        past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
        if self.args.ztype == 'gaussian':
            pz_distribution = Normal(
                mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device),
                logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_feature.device))
        else:
            ValueError('Unknown hidden distribution!')
        pz_sampled = pz_distribution.rsample()
        z = pz_sampled
        node_past = nodes_current[:obs_length].transpose(0, 1)
        diverse_pred_traj, _ = self.decoder(past_feature_repeat, z, node_past, sample_num=self.args.sample_num,
                                            mode='inference')
        #  (agent_num, sample_num, self.past_length, 2) -> (sample_num,agent_num,  self.past_length, 2)
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
        if self.args.phase == 'test' and self.args.vis == 'sne':
            return diverse_pred_traj, target_pred_traj,past_feature
        else:
            return diverse_pred_traj, target_pred_traj

# 无用的代码 错误的写法
class NEW_STAR(torch.nn.Module):

    def __init__(self, args, dropout_prob=0):
        super(NEW_STAR, self).__init__()

        # set parameters for network architecture
        self.embedding_size = 32
        self.output_size = 2
        self.dropout_prob = dropout_prob
        self.args = args
        self.mean = []
        self.var = []

        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward network ModelStrategy in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value

        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        """
        舍弃无效的self.temporal_encoder_layer 后续直接添加 避免梯度为NONE
        self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)
        self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
        self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer, 1)
        """
        self.temporal_encoder_1 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        self.temporal_encoder_2 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)

        # Linear layer to output and fusion
        # inplace-operation 问题点 最终显示该变量version变化，即需要version-7，但后续变量已经被修改了 成version——9
        self.output_layer = nn.Linear(48, 2)
        self.fusion_layer = nn.Linear(64, 32)
        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(self.dropout_prob)
        self.dropout_in2 = nn.Dropout(self.dropout_prob)
        # CVAE
        self.encoder_past_layer = nn.Linear(8, 1)
        # todo 可能是 维度需要注意设计
        self.encoder_furture_layer = nn.Linear(11, 1)
        self.out_mlp = MLP2(input_dim=64, hidden_dims=[32], activation='relu')
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        self.pz_layer = nn.Linear(self.embedding_size, 2 * self.args.zdim)
        self.decoder = Decoder(args)
        """
        inplace
        ReLU uses its output for the gradient computation as defined here and as shown in this code snippet:
        故而relu在使用时需要注意其 inplace为true或false true节省开销 但容易造成inplace operation
        """

    def get_st_ed(self, batch_num):
        """

        :param batch_num: contains number of pedestrians in different scenes for a batch
        :type batch_num: list
        :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
        :rtype: list
        """
        cumsum = torch.cumsum(batch_num, dim=0)
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

        st_ed.insert(0, (0, int(cumsum[0])))

        return st_ed

    def get_node_index(self, seq_list):
        """

        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):

            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices

    def update_batch_pednum(self, batch_pednum, ped_list):
        """

        :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
        :type list
        :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
        :type tensor
        :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
        :rtype: list
        """
        updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
        updated_batch_pednum = copy.deepcopy(batch_pednum)

        cumsum = np.cumsum(updated_batch_pednum_)
        new_ped = copy.deepcopy(ped_list).cpu().numpy()

        for idx, num in enumerate(cumsum):
            num = int(num)
            if idx == 0:
                updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
            else:
                updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

        return updated_batch_pednum

    def mean_normalize_abs_input(self, node_abs, st_ed):
        """

        :param node_abs: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: node_abs: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        node_abs = node_abs.permute(1, 0, 2)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)

    def calculate_loss_kl(self, qz_distribution, pz_distribution, batch_ped, min_clip):
        loss = qz_distribution.kl(pz_distribution).sum()
        loss /= (batch_ped)
        loss_clamp = loss.clamp_min_(min_clip)
        return loss_clamp

    def forward(self, inputs, iftest=False):
        # 分析：最小破坏结构的做法即前8s的时间进行Encoder；而后相应的在Temporal Transformer处修改代码
        # 这样前8s每个时刻点捕获到了空间关系，而后又有时间关系的捕捉 继而加入CVAE与Decoder架构

        # ifmeta-test用来表征相应的是否需要进行注入操作，而相应的mean-list以及var——list则表示的是对应的不同域的特征均值和方差
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]
        # 后续可能需要改shape的19为8（前8sencoder）不需要
        # outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

        # noise = get_noise((1, 16), 'gaussian')
        #  todo temporal——transformer 将一个人轨迹截止当前frame时刻的数据输进去，而后相应的输出重构后的该人历史轨迹数据；
        #   相当于重新编码了过去的轨迹，故而此处的encoder最后得到结果应该是【8，batch，hidden_dim】而后再将这8通过FC？转为1，
        #   当做过去轨迹的特征，相应的特征混合也应该在此处进行混合 此前混合单个特征 无法反映特征所包含的信息
        # encoder_var = 0
        # 因为要用CVAE框架，那么过去与未来就都需要编码，相应的，此处则全部由循环编码过程完成，无上飞线，有GM线
        for framenum in range(self.args.seq_length - 1):
            # todo 需要注意的是add to history 只有在obs之后进行
            # node-idx筛选出从起始到当前帧都存在的ped
            node_index = self.get_node_index(seq_list[:framenum + 1])
            nei_list = nei_lists[framenum, node_index, :]
            nei_list = nei_list[:, node_index]
            # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；仍然会随着frame的变换而变换
            updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
            # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
            st_ed = self.get_st_ed(updated_batch_pednum)
            # 只取到framenum的数据
            nodes_current = nodes_norm[:framenum + 1, node_index]
            # We normalize the absolute coordinates using the mean value in the same scene
            # 基于同一场景中的行人数据进行标准化，即运用该windows所有行人的平均xy坐标进行分析
            node_abs = self.mean_normalize_abs_input(nodes_abs[:framenum + 1, node_index], st_ed)
            # Input Embedding
            # 此处作用将输入的xy 2维坐标变量转换为32维向量，相应的先linear，后relu，而后dropout；此处无inplace=True问题，也无+=问题；
            if framenum == 0:
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
            else:
                # 当framenum=18时，相应的nodes——current为【19,140,2】
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
                # GM对应论文中的Graph Memory
                temporal_input_embedded = temporal_input_embedded.clone()
                temporal_input_embedded[:framenum] = GM[:framenum, node_index]

            # 需要注意 temporal（nodes-current：经过基于obs观测帧的归一化）和spatial（node-abs基于全局的norm）输入的node序列数据经过不同的处理，
            spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs)))
            # 数据流式处理，空间输入基于最新的一帧进行分析，过往的空间关系不考虑
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)

            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
            # 时间输入基于完整的截止当前帧的数据（但是其中只有最新帧的数据是输入的，最新帧往前的数据是基于过往的预测结果的，而不是原始的结果）输出会重构所有帧数据，但只取最后一帧
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)[-1]
            # 取temporal-input-embedded初始到倒数第二个数据，此处倒数第一个数据经过encoder后被重构！！
            temporal_input_embedded = temporal_input_embedded[:-1]
            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)
            # 经过第一次encoder后的最新帧数据都变了，对其拼接，输入spatial-encoder
            spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)
            # 将经过spatial_encoder_2的数据与原先的temporal_input_embedded（初始到倒数第二个）进行拼接：正好又拼接成完整的序列
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            # todo 重构过后的完整历史轨迹特征
            # encoder_var = self.temporal_encoder_2(temporal_input_embedded)
            # temporal_input_embedded = encoder_var[-1]
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)[-1]
            # noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
            # temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)
            # outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            # todo 回传 outputs：相应的将预测结果传回(只在obs之后开始回传)，在预测新的一帧时运用，GM则回传中间特征层的数据，每次都只回传最新帧
            # outputs[framenum, node_index] = outputs_current
            # todo是否是因为此处还未backwardtemporal_input_embedded，而GM的值就已经改变，而使得对应的值也被变了
            GM[framenum, node_index] = temporal_input_embedded
        # Reshape input to (8, batch——size，hidden_dim) 转变为(batch——size，hidden_dim,8)（batch——size*hidden——dim，8）
        encoder_past = GM[:self.args.obs_length, :, :].permute(1, 2, 0)
        encoder_furture = GM[self.args.obs_length:, :, :].permute(1, 2, 0)
        encoder_past = encoder_past.view(-1, encoder_past.size(-1))
        encoder_furture = encoder_furture.view(-1, encoder_furture.size(-1))
        # Pass through the fully connected layer Reshape back to ( batch_size, hidden_dim)
        encoder_past = self.encoder_past_layer(encoder_past).view(-1, self.embedding_size)
        encoder_furture = self.encoder_furture_layer(encoder_furture).view(-1, self.embedding_size)

        # ---------------------CVAE-------------------------------
        ### q dist ###
        """
        根据超参数self.args.ztype的设置，选择创建哪种分布类型的后验分布q(z|x,y)。如果ztype为'gaussian'，
        则使用Normal（高斯分布）类创建一个正态分布，并以qz_param作为分布的参数;
        从后验分布q(z | x,y)中抽样得到qz_sampled，将其用于计算KL散度（KL divergence）损失项。
        """
        # (batch_size,hidden_dim*2) 64
        h = torch.cat((encoder_past, encoder_furture), dim=-1)
        # (batch_size,16) 64->16
        h = self.out_mlp(h)
        # 在变分自编码器中，潜变量 z 的均值和方差是通过编码器网络输出的，并且需要满足一定的分布假设，例如高斯分布。
        # 因此，该线性层的作用是将 MLP 的输出映射到满足分布假设的潜变量均值和方差，从而使得潜变量 z 可以被正确地解码和重构。
        qz_param = self.qz_layer(h)
        if self.args.ztype == 'gaussian':
            qz_distribution = Normal(params=qz_param)
        else:
            ValueError('Unknown hidden distribution!')
        qz_sampled = qz_distribution.rsample()
        ### p dist ###
        """
        根据超参数self.args.learn_prior的设置，选择创建哪种分布类型的先验分布p(z)。如果learn_prior为True，
        则使用线性层对象self.pz_layer生成一个包含均值和标准差参数的向量pz_param，并以此创建一个Normal分布pz_distribution。
        如果learn_prior为False，则以0为均值、方差为1的标准正态分布作为先验分布p(z)
        """
        if self.args.learn_prior:
            pz_param = self.pz_layer(encoder_past)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=pz_param)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(encoder_past.shape[0], self.args.zdim).to(encoder_past.device),
                                         logvar=torch.zeros(encoder_past.shape[0], self.args.zdim).to(
                                             encoder_past.device))
            else:
                ValueError('Unknown hidden distribution!')
        ### use q ###
        # z = qz_sampled
        # 基于解码器，输入过去轨迹past_traj()和特征past_feature()，采样的z，agent_num
        pred_traj, recover_traj = self.decoder(encoder_past, qz_sampled, nodes_abs, sample_num=1)
        output_seq = torch.cat((recover_traj, pred_traj), dim=1)
        # (batch,20,2) -> (20,batch,2)
        output_seq = output_seq.transpose(0, 1)
        batch_pednum = encoder_past.shape[0]
        loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_pednum, self.args.min_clip)

        return output_seq, loss_kl

    def inference(self, data):
        # 等待开发

        return None