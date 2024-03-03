import numpy as np
import torch
import torch.nn as nn
from Model.CVAE_utils import Normal, MLP2
from torch.distributions.normal import Normal as Normal_official
from Model.star import TransformerModel, TransformerEncoder, TransformerEncoderLayer
from Model.star_cvae import Decoder, STAR_CVAE

torch.manual_seed(0)

""" Positional Encoding """


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        # 矩阵将被用来存储每个位置的编码。
        pe = torch.zeros(max_len, self.d_model)
        # 生成一个从0到max_len - 1的一维张量，代表序列中每个位置的索引，然后通过 unsqueeze(1) 将其变为二维张量（列向量）。这使得每一行对应一个位置索引。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        """
        计算了位置编码的分母项。
        首先，使用 torch.arange(0, self.d_model, 2) 生成一个从0开始，步长为2的序列，这样就只考虑了偶数索引（因为我们将使用正弦和余弦交替）。
        然后，这个序列通过乘以 (-np.log(10000.0) / self.d_model) 并应用指数函数 exp 来变换，得到分母项。
        """
        # 先log下来 在exp上去 同时因为分母是2i
        # PE（pos,2i）   = sin(pos/(10000)^(2i/d_model))
        # PE (pos,2i+1) = cos(pos/(10000)^(2i/d_model))
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        # 分别使用正弦和余弦函数来填充位置编码矩阵 pe。对于矩阵中的每一行（代表不同的位置索引），在偶数索引位置使用正弦函数，奇数索引位置使用余弦函数
        pe[:, 0::2] = torch.sin(position * div_term) # 从索引 0 开始，每隔2个元素选取一个元素
        pe[:, 1::2] = torch.cos(position * div_term) # 从索引 1 开始，每隔2个元素选取一个元素
        return pe

    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.unsqueeze(1).repeat(1, num_a, 1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        # 输入维度 [len, num_ped, 32]
        num_t = x.shape[0]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)  # (T,N,D)
        # 此处拼接x和位置编码
        # 如果 concat 为真，则将输入特征和位置编码拼接，并通过全连接层。否则，直接将位置编码加到输入特征上。
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            # 拼接完成后 继续经过 MLP
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x)  # (N,T,D)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class STEncoder_HIN(nn.Module):
    """
    输入的是past相关的数据，经过对应的位置编码，HIN，空时间双Transformer架构
    """

    def __init__(self, args, stage, dropout_prob=0):
        super(STEncoder_HIN, self).__init__()
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
        # ！！ HIN专有的混合不同类型的边的注意力的方法 node-level  semantic-level
        self.semantic_attention_1 = SemanticAttention(in_size=emsize)
        self.semantic_attention_2 = SemanticAttention(in_size=emsize)
        # ！！新添加的pos-encoder
        self.pos_encoder1 = PositionalAgentEncoding(emsize, 0.1, concat=True)
        self.pos_encoder2 = PositionalAgentEncoding(emsize, 0.1, concat=True)

        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)
        # Linear layer to output and fusion
        self.fusion_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)
        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in_1 = nn.Dropout(self.dropout_prob)
        self.dropout_in_2 = nn.Dropout(self.dropout_prob)
        # 注意相对应的ST-HIN采用的是双路的结构 并且用的都是经过transformer后的最后一步的值 故而此处用不到full-layer 需要注释掉 以防其在MLDG的过程中发生错误
        if stage == 'past':
            self.encoder_full_layer = nn.Linear(self.args.obs_length, 1)
        elif stage == 'future':
            self.encoder_full_layer = nn.Linear(self.args.pred_length, 1)

    def forward(self, inputs):
        # nodes_current/abs (8,num_ped,2),nei_list(relation_num, 8, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]
        num_ped = nodes_current.shape[1]
        # todo 新的不同结构 空间时间 时间空间双层结构
        # 时间-空间分支
        # 时间编码 2- 32 drop transformer [len,num_ped,32]
        # temporal_input_embedded = self.dropout_in_1(self.relu(self.input_embedding_layer_temporal(nodes_current)))
        nodes_current_embedded = self.input_embedding_layer_temporal(nodes_current)
        nodes_current_pos = self.pos_encoder1(nodes_current_embedded, num_a=num_ped)
        temporal_input_embedded = self.dropout_in_1(self.relu(nodes_current_pos))

        temporal_input_embedded_1 = self.temporal_encoder_1(temporal_input_embedded)
        # todo nei——list 多添加一个维度 对应为关系的数量 ！！ 注意此处的数据处理需要对应的更改
        spatial_middle_relation_embedded = []
        # 求解HIN内部各自的node-level的注意力 汇聚数据
        for relationnum in range(nei_list.shape[0]):
            # 注意此处的spatial-encoder-1只接受一帧，此处接受的是temporal的最后一帧数据//最后一帧的embedding以及相应的nei-list todo 认为不需要全部过一遍？
            spatial_middle_embedded = self.spatial_encoder_1(temporal_input_embedded_1[-1].unsqueeze(1),
                                                             nei_list[relationnum][-1])
            spatial_middle_relation_embedded.append(spatial_middle_embedded)
        spatial_middle_relation_embedded_ = torch.cat(spatial_middle_relation_embedded, axis=1)
        # semantic-level 汇聚
        spatial_feat_embedded = self.semantic_attention_1(spatial_middle_relation_embedded_)
        spatial_feat_embedded = spatial_feat_embedded
        # 空间-时间分支
        #spatial_input_embedded = self.dropout_in_2(self.relu(self.input_embedding_layer_spatial(nodes_abs_position)))
        nodes_abs_position_embedded = self.input_embedding_layer_spatial(nodes_abs_position)
        nodes_abs_position_pos = self.pos_encoder2(nodes_abs_position_embedded, num_a=num_ped)
        spatial_input_embedded = self.dropout_in_2(self.relu(nodes_abs_position_pos))

        spatial_embedded_2 = torch.zeros(length, num_ped, self.embedding_size).cuda()
        # 逐帧处理 空间结构
        for frame in range(length):
            spatial_embedded_2_tmp = spatial_input_embedded[frame]
            spatial_input_relation_embedded_2 = []
            for relationnum in range(nei_list.shape[0]):
                spatial_middle_embedded_2 = self.spatial_encoder_2(spatial_embedded_2_tmp.unsqueeze(1),
                                                                   nei_list[relationnum][frame])
                spatial_input_relation_embedded_2.append(spatial_middle_embedded_2)
            # 合并semantic level
            spatial_input_relation_embedded_2 = torch.cat(spatial_input_relation_embedded_2, axis=1)
            spatial_middle_embedded = self.semantic_attention_2(spatial_input_relation_embedded_2)
            spatial_middle_embedded = spatial_middle_embedded.unsqueeze(0)
            spatial_embedded_2[frame] = spatial_middle_embedded
        # 经过temporal-encoder后获得对应的last frame
        temporal_feature_embedded_last = self.temporal_encoder_2(spatial_embedded_2)[-1]
        # 拼接双路的frame结构 并fusion !
        fusion_feat = torch.cat((spatial_feat_embedded, temporal_feature_embedded_last), dim=1)
        fusion_feat = self.fusion_layer(fusion_feat)
        return fusion_feat


class STAR_CVAE_HIN(STAR_CVAE):
    def __init__(self, args):
        super(STAR_CVAE_HIN, self).__init__(args)
        self.args = args
        self.embedding_size = 32
        # modeule structure
        self.past_encoder = STEncoder_HIN(args, stage='past')
        self.future_encoder = STEncoder_HIN(args, stage='future')
        self.out_mlp = MLP2(input_dim=64, hidden_dims=[32], activation='relu')
        # todo 此处维度需要注意 可能有问题 后期调节超参数
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        # todo 注意此处结合分析pz-layer 即学习先验的必要性
        # self.pz_layer = nn.Linear(self.embedding_size, 2*self.args.zdim)
        self.decoder = Decoder(args)

    def forward(self, inputs, stage, mean_list=[], var_list=[], ifmixup=False):
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 注意 此处针对的是不同帧下行人数量不一的问题，传统的解决思路是直接提取从头到尾到存在的行人，而star方法则是逐帧中提取从头到尾到存在的行人，
        # 最后的loss计算时也是只考虑从头到尾的行人，这个只是相当于利用了更多行人的信息；但在此处，我们采用传统思路预处理，因为我们是针对完全形态分析
        # node-idx筛选出从起始到当前帧都存在的ped
        # todo 此处的nei-list需要重写 相应的因为多了一个维度  可能相应的后续seq-list，nei-num，batch-pednum都可能需要再改写
        node_index = self.get_node_index(seq_list)
        # nei_list = nei_lists[:, node_index, :]
        # nei_list = nei_list[:, :, node_index]
        # ！！ nei-list 形状为 relation frame num-ped ，num-ped
        nei_list = nei_lists[:, :, node_index, :]
        nei_list = nei_list[:, :, :, node_index]

        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        # todo 在batch41出现batch_pednum = 0 的情况 ！
        if batch_pednum.cpu().detach().numpy().shape[0] - updated_batch_pednum.cpu().detach().numpy().shape[0] > 0:
            print(
                'batch_pednum:' + str(batch_pednum.cpu().detach().numpy().shape) + '/' + 'updated_batch_pednum:' + str(
                    batch_pednum.cpu().detach().numpy().shape))

        st_ed = self.get_st_ed(updated_batch_pednum)
        # todo 提取新的轨迹数据 提取的是未归一化的
        nodes_current = nodes_abs[:, node_index, :]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_abs[:, node_index], st_ed)
        # todo 注意 nei-list形状
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:, :obs_length]
        future_traj = nodes_current[obs_length:], nodes_abs_position[obs_length:], nei_list[:, obs_length:]
        # todo ----------修改在上面
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
        if self.args.ztype == 'gaussian':
            # todo 分析出 mu logvar sigma 用法
            pz_distribution = Normal(mu=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_feature.device),
                                     logvar=torch.zeros(past_feature.shape[0], self.args.zdim).to(
                                         past_feature.device))
        else:
            ValueError('Unknown hidden distribution!')

        ### use q ###
        # z = qz_sampled 基于解码器，输入过去轨迹past_traj()和特征past_feature()，采样的z，agent_num
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
        # nei_list = nei_lists[:, node_index, :]
        # nei_list = nei_list[:, :, node_index]
        # todo 基于相应的不同的 nei-list进行设计的
        nei_list = nei_lists[:, :, node_index, :]
        nei_list = nei_list[:, :, :, node_index]
        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index, :]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_abs[:, node_index], st_ed)
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:, :obs_length]
        past_feature = self.past_encoder(past_traj)
        target_pred_traj = nodes_current[obs_length:] # 基于非归一化的轨迹
        # 上述代码计算潜在空间中的特征向量past_feature，相同于forward，只是在测试推理过程中不需要未来数据
        sample_num = 20
        # todo pz-layer
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
        return diverse_pred_traj, target_pred_traj

    def inference_visual(self, inputs,scene_frame):
        # scene_frame 包含对应的scene和frame数据
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
        new_scene_frame = scene_frame.loc[node_index.cpu().numpy(),:]
        # 基于node——index取出数据获得新的scene-frame以及新的inputs 从而确保过去的轨迹正确对应
        # nei_list = nei_lists[:, node_index, :]
        # nei_list = nei_list[:, :, node_index]
        # todo 基于相应的不同的 nei-list进行设计的
        nei_list = nei_lists[:, :, node_index, :]
        nei_list = nei_list[:, :, :, node_index]
        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index, :]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_abs[:, node_index], st_ed)
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:, :obs_length]
        past_feature = self.past_encoder(past_traj)
        target_pred_traj = nodes_current[obs_length:] # 基于非归一化的轨迹
        # 上述代码计算潜在空间中的特征向量past_feature，相同于forward，只是在测试推理过程中不需要未来数据
        sample_num = 20
        # todo pz-layer
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
        # 返回完整的轨迹样式，因为SDD中进行了数据删除 不在对应了
        return diverse_pred_traj, nodes_current,new_scene_frame
