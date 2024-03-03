import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.CVAE_utils import Normal, MLP2, MLP
from torch.distributions.normal import Normal as Nomal_official
from Model.star import TransformerModel, TransformerEncoder, TransformerEncoderLayer
from Model.star_cvae import Decoder, STAR_CVAE

torch.manual_seed(0)


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
        pe[:, 0::2] = torch.sin(position * div_term)  # 从索引 0 开始，每隔2个元素选取一个元素
        pe[:, 1::2] = torch.cos(position * div_term)  # 从索引 1 开始，每隔2个元素选取一个元素
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


"""
分析区别而后理解在空间与时间模块处分别应该用哪些：
TransformerModel：仍然只有编码器部分，但是只是更多的封装了源序列掩码（src_mask）
TransformerEncoder：由多个 TransformerEncoderLayer 实例组成，它的 forward 方法依次将输入数据 src 传递给每一个编码器层
TransformerEncoderLayer：完成transformer内完整的单独一层encoder；它首先执行多头注意力，然后是前馈网络，每个子层后都跟有残差连接和层归一化。
"""


# 完整的IT,TI 两个branch结构 整体模型
class Dual_TT_Encoder(nn.Module):
    """
    包含对应的IM和MI两条路径
    同时最好可以在这个版本进行消融实验的选择分析！
    初版分析:
    """
    #dropout_prob=0 需要注意！
    def __init__(self, args, stage, dropout_prob=0):
        super(Dual_TT_Encoder, self).__init__()
        self.embedding_size = 32
        self.dropout_prob = dropout_prob
        self.args = args
        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward network ModelStrategy in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value
        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)
        # Linear layer to map different path TI/IT
        self.input_TI_layer = nn.Linear(32, 32)
        self.input_IT_layer = nn.Linear(32, 32)
        # Linear layer to output and fusion
        self.fusion_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)
        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in_1 = nn.Dropout(self.dropout_prob)
        self.dropout_in_2 = nn.Dropout(self.dropout_prob)
        # 为MLP服务
        self.dropout_in_TI = nn.Dropout(self.dropout_prob)
        self.dropout_in_IT = nn.Dropout(self.dropout_prob)
        # 后续的消融也是基于下文的两个空间两个时间进行分析
        # 空间模块基于transformerModel 其中的mask为对应的nei-lists即邻接矩阵
        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        # 时间模块 此处的输入数据其实只保留了整个时间序列都存在的序列 故而可知的是不用mask 并且此处是基于1层的
        self.temporal_encoder_1 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        self.temporal_encoder_2 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        # PE函数 对应于两个时间transformer
        self.pos_encoder1 = PositionalAgentEncoding(emsize, 0.1, concat=True)
        self.pos_encoder2 = PositionalAgentEncoding(emsize, 0.1, concat=True)
        # 涉及到如何将过去8帧的数据或则说是未来的12帧的数据进行合并？？
        # 直接选取最后一帧是常见做法，或则说再过一个linear或max-pooling
        # 注意相对应的ST-HIN采用的是双路的结构 并且用的都是经过transformer后的最后一步的值 故而此处用不到full-layer 需要注释掉 以防其在MLDG的过程中发生错误
        # if stage == 'past':
        #    self.encoder_full_layer = nn.Linear(self.args.obs_length, 1)
        # elif stage == 'future':
        #    self.encoder_full_layer = nn.Linear(self.args.pred_length, 1)

    def forward(self, inputs):
        # nodes_current(未有空间或时间归一化)/abs(基于每个场景进行过空间归一化) (length,num_ped,2),nei_list(length, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]  # pred-length;obs-length;
        num_ped = nodes_current.shape[1]
        # todo 先完成一版未基于时间归一化的数据 模型处理
        # TI branch 先时间后空间
        # temporal embedding
        nodes_current_embedded = self.input_embedding_layer_temporal(nodes_current)
        if self.args.PE == 'True':
            nodes_current_pos = self.pos_encoder1(nodes_current_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_current_pos = nodes_current_embedded
        temporal_input_embedded_origin = self.dropout_in_1(self.relu(nodes_current_pos))
        # TM branch
        temporal_input_embedded_temporal_TS = self.temporal_encoder_1(temporal_input_embedded_origin)
        # X+MLP(TM(X)) todo => 分析此处的X应该是具体何值？高维映射 不然2和32维度无法对应的添加
        temporal_input_embedded_temporal_TS = temporal_input_embedded_origin + self.dropout_in_TI(
            self.relu(self.input_TI_layer(temporal_input_embedded_temporal_TS)))
        # 此处直接用cuda 不知是否需要单独指定device
        # SM branch
        spatial_input_embedded_spatial_TS = torch.zeros(length, num_ped, self.embedding_size).cuda()
        for frame in range(length):
            # [length,num-ped,32]->[num-ped,32]->[num_ped,1,32]
            spatial_input_embedded_spatial_frame_TS = self.spatial_encoder_1(
                temporal_input_embedded_temporal_TS[frame].unsqueeze(1), nei_list[frame])
            # [num_ped,1,32]->[1,num_ped,32]->[num_ped,32]
            spatial_input_embedded_spatial_frame_TS = spatial_input_embedded_spatial_frame_TS.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_TS[frame] = spatial_input_embedded_spatial_frame_TS
        TS_embedding = spatial_input_embedded_spatial_TS

        # IT分子 先空间后时间
        # Spatial embedding
        nodes_abs_position_embedded = self.input_embedding_layer_spatial(nodes_abs_position)
        if self.args.PE == 'True':
            # nodes_abs_position_pos = nodes_abs_position_embedded # 为了验证对应的空间Encoder的有效性
            nodes_abs_position_pos = self.pos_encoder2(nodes_abs_position_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_abs_position_pos = nodes_abs_position_embedded
        spatial_input_embedded_origin = self.dropout_in_2(self.relu(nodes_abs_position_pos))
        # SM branch
        spatial_input_embedded_spatial_ST = torch.zeros(length, num_ped, self.embedding_size).cuda()
        for frame in range(length):
            spatial_input_embedded_spatial_frame_ST = self.spatial_encoder_2(
                spatial_input_embedded_origin[frame].unsqueeze(1), nei_list[frame])
            spatial_input_embedded_spatial_frame_ST = spatial_input_embedded_spatial_frame_ST.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_ST[frame] = spatial_input_embedded_spatial_frame_ST
        # X+MLP(SM(X)) todo
        spatial_input_embedded_spatial_ST = spatial_input_embedded_origin + self.dropout_in_IT(
            self.relu(self.input_IT_layer(spatial_input_embedded_spatial_ST)))
        # TM branch
        temporal_input_embedded_temporal_ST = self.temporal_encoder_2(spatial_input_embedded_spatial_ST)
        ST_embedding = temporal_input_embedded_temporal_ST
        # 拼接双路结构 todo => MaxPooling?
        # 计算对齐Loss =》 依据余弦相似度
        # 计算A和B之间的余弦相似度
        cosine_sim = F.cosine_similarity(TS_embedding[-1], ST_embedding[-1], dim=1)
        # 定义损失函数为1减去余弦相似度的平均值
        loss = 1 - cosine_sim.mean()
        # 直接拼接最后一维 即【num_ped,dim】
        fusion_feat_origin = torch.cat((TS_embedding[-1], ST_embedding[-1]), dim=1)
        # 拼接双路的frame结构 并fusion !
        fusion_feat = self.fusion_layer(fusion_feat_origin)
        return fusion_feat,loss


class Dual_TI_Encoder(Dual_TT_Encoder):
    def __init__(self, args,stage):
        super().__init__(args=args,stage=stage)
        assert self.args.Dual_TT_ablation == "TI"
        print("backbone消融实验选取" + self.args.Dual_TT_ablation + "即单独一路TI")

    def forward(self, inputs):
        # 单独一路
        # nodes_current(未有空间或时间归一化)/abs(基于每个场景进行过空间归一化) (length,num_ped,2),nei_list(length, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]  # pred-length;obs-length;
        num_ped = nodes_current.shape[1]
        # todo 先完成一版未基于时间归一化的数据 模型处理
        # TI branch 先时间后空间
        # temporal embedding
        nodes_current_embedded = self.input_embedding_layer_temporal(nodes_current)
        if self.args.PE == 'True':
            nodes_current_pos = self.pos_encoder1(nodes_current_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_current_pos = nodes_current_embedded
        temporal_input_embedded_origin = self.dropout_in_1(self.relu(nodes_current_pos))
        # TM branch
        temporal_input_embedded_temporal_TS = self.temporal_encoder_1(temporal_input_embedded_origin)
        # X+MLP(TM(X)) todo => 分析此处的X应该是具体何值？高维映射 不然2和32维度无法对应的添加
        # temporal_input_embedded_temporal_TS = temporal_input_embedded_origin + self.dropout_in_TI(self.relu(self.input_TI_layer(temporal_input_embedded_temporal_TS)))
        # 此处直接用cuda 不知是否需要单独指定device
        # SM branch
        spatial_input_embedded_spatial_TS = torch.zeros(length, num_ped, self.embedding_size).cuda()
        for frame in range(length):
            # [length,num-ped,32]->[num-ped,32]->[num_ped,1,32]
            spatial_input_embedded_spatial_frame_TS = self.spatial_encoder_1(
                temporal_input_embedded_temporal_TS[frame].unsqueeze(1), nei_list[frame])
            # [num_ped,1,32]->[1,num_ped,32]->[num_ped,32]
            spatial_input_embedded_spatial_frame_TS = spatial_input_embedded_spatial_frame_TS.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_TS[frame] = spatial_input_embedded_spatial_frame_TS
        TS_embedding = spatial_input_embedded_spatial_TS
        return TS_embedding[-1]


class Dual_IT_Encoder(Dual_TT_Encoder):
    def __init__(self, args, stage):
        super().__init__(args=args, stage=stage)
        assert self.args.Dual_TT_ablation == "IT"
        print("backbone消融实验选取" + self.args.Dual_TT_ablation + "即单独一路IT")

    def forward(self, inputs):
        # nodes_current(未有空间或时间归一化)/abs(基于每个场景进行过空间归一化) (length,num_ped,2),nei_list(length, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]  # pred-length;obs-length;
        num_ped = nodes_current.shape[1]
        # IT分子 先空间后时间
        # Spatial embedding
        nodes_abs_position_embedded = self.input_embedding_layer_spatial(nodes_abs_position)
        if self.args.PE == 'True':
            nodes_abs_position_pos = self.pos_encoder2(nodes_abs_position_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_abs_position_pos = nodes_abs_position_embedded
        spatial_input_embedded_origin = self.dropout_in_2(self.relu(nodes_abs_position_pos))
        # SM branch
        spatial_input_embedded_spatial_ST = torch.zeros(length, num_ped, self.embedding_size).cuda()
        for frame in range(length):
            spatial_input_embedded_spatial_frame_ST = self.spatial_encoder_2(
                spatial_input_embedded_origin[frame].unsqueeze(1), nei_list[frame])
            spatial_input_embedded_spatial_frame_ST = spatial_input_embedded_spatial_frame_ST.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_ST[frame] = spatial_input_embedded_spatial_frame_ST
        # X+MLP(SM(X)) todo
        # spatial_input_embedded_spatial_ST = spatial_input_embedded_origin + self.dropout_in_IT(self.relu(self.input_IT_layer(spatial_input_embedded_spatial_ST)))
        # TM branch
        temporal_input_embedded_temporal_ST = self.temporal_encoder_2(spatial_input_embedded_spatial_ST)
        ST_embedding = temporal_input_embedded_temporal_ST
        return ST_embedding[-1]


class Dual_II_Encoder(Dual_TT_Encoder):
    def __init__(self, args, stage):
        super().__init__(args=args, stage=stage)
        assert self.args.Dual_TT_ablation == "II"
        print("backbone消融实验选取" + self.args.Dual_TT_ablation + "即单独一路InteractiveI")

    def forward(self, inputs):
        # nodes_current(未有空间或时间归一化)/abs(基于每个场景进行过空间归一化) (length,num_ped,2),nei_list(length, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]  # pred-length;obs-length;
        num_ped = nodes_current.shape[1]
        # IT分子 先空间后时间
        # Spatial embedding
        nodes_abs_position_embedded = self.input_embedding_layer_spatial(nodes_abs_position)
        if self.args.PE == 'True':
            nodes_abs_position_pos = self.pos_encoder2(nodes_abs_position_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_abs_position_pos = nodes_abs_position_embedded
        spatial_input_embedded_origin = self.dropout_in_2(self.relu(nodes_abs_position_pos))
        # SM branch 1
        spatial_input_embedded_spatial_ST_1 = torch.zeros(length, num_ped, self.embedding_size).cuda()
        for frame in range(length):
            spatial_input_embedded_spatial_frame_ST_1 = self.spatial_encoder_1(
                spatial_input_embedded_origin[frame].unsqueeze(1), nei_list[frame])
            spatial_input_embedded_spatial_frame_ST_1 = spatial_input_embedded_spatial_frame_ST_1.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_ST_1[frame] = spatial_input_embedded_spatial_frame_ST_1
        # SM branch 2
        spatial_input_embedded_spatial_ST_2 = torch.zeros(length, num_ped, self.embedding_size).cuda()
        for frame in range(length):
            spatial_input_embedded_spatial_frame_ST_2 = self.spatial_encoder_2(
                spatial_input_embedded_spatial_ST_1[frame].unsqueeze(1), nei_list[frame])
            spatial_input_embedded_spatial_frame_ST_2 = spatial_input_embedded_spatial_frame_ST_2.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_ST_2[frame] = spatial_input_embedded_spatial_frame_ST_2

        return spatial_input_embedded_spatial_ST_2[-1]


class Dual_TemporalT_Encoder(Dual_TT_Encoder):
    def __init__(self, args, stage):
        super().__init__(args=args, stage=stage)
        assert self.args.Dual_TT_ablation == "TT"
        print("backbone消融实验选取" + self.args.Dual_TT_ablation + "即单独一路TemporalTemporal")

    def forward(self, inputs):
        # 单独一路
        # nodes_current(未有空间或时间归一化)/abs(基于每个场景进行过空间归一化) (length,num_ped,2),nei_list(length, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]  # pred-length;obs-length;
        num_ped = nodes_current.shape[1]
        # todo 先完成一版未基于时间归一化的数据 模型处理
        # TI branch 先时间后空间
        # temporal embedding
        nodes_current_embedded = self.input_embedding_layer_temporal(nodes_current)
        if self.args.PE == 'True':
            nodes_current_pos = self.pos_encoder1(nodes_current_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_current_pos = nodes_current_embedded
        temporal_input_embedded_origin = self.dropout_in_1(self.relu(nodes_current_pos))
        # TM branch 1
        temporal_input_embedded_temporal_TM_1 = self.temporal_encoder_1(temporal_input_embedded_origin)
        # TM branch 2
        temporal_input_embedded_temporal_TM_2 = self.temporal_encoder_2(temporal_input_embedded_temporal_TM_1)
        return temporal_input_embedded_temporal_TM_2[-1]


# 为消融实验准备的套壳
def Dual_TT_Encoder_Select(args=None, stage='past'):
    if args.Dual_TT_ablation == "Dual_TT":
        return Dual_TT_Encoder(args, stage=stage)
    elif args.Dual_TT_ablation == "IT":
        return Dual_IT_Encoder(args, stage=stage)
    elif args.Dual_TT_ablation == 'TI':
        return Dual_TI_Encoder(args, stage=stage)
    elif args.Dual_TT_ablation == 'II':
        return Dual_II_Encoder(args, stage=stage)
    elif args.Dual_TT_ablation == 'TT':
        return Dual_TemporalT_Encoder(args, stage=stage)
    else:
        raise NotImplementedError


class Dual_TT(STAR_CVAE):
    def __init__(self, args):
        super(Dual_TT, self).__init__(args)
        self.args = args
        self.embedding_size = 32
        # model structure
        self.past_encoder = Dual_TT_Encoder_Select(args, stage='past')
        self.future_encoder = Dual_TT_Encoder_Select(args, stage='future')
        self.out_mlp = MLP2(input_dim=64, hidden_dims=[32], activation='relu')
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        self.pz_layer = nn.Linear(self.embedding_size, 2 * self.args.zdim)
        self.decoder = Decoder(args)

    def forward(self, inputs, stage):
        # 此处的stage是对应的support的
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259)
        # nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 预处理数据只考虑从头到尾的行人，这个只是相当于利用了更多行人的信息；但在此处，我们采用传统思路预处理，因为我们是针对完全形态分析
        # node-idx筛选出从起始到当前帧都存在的ped
        node_index = self.get_node_index(seq_list)
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]
        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        if batch_pednum.cpu().detach().numpy().shape[0] - updated_batch_pednum.cpu().detach().numpy().shape[0] > 0:
            print(
                'batch_pednum:' + str(batch_pednum.cpu().detach().numpy().shape) + '/' + 'updated_batch_pednum:' + str(
                    batch_pednum.cpu().detach().numpy().shape))
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_current, st_ed)
        # 输入encoder的前置知识
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:obs_length]
        future_traj = nodes_current[obs_length:], nodes_abs_position[obs_length:], nei_list[obs_length:]
        # (num_ped,hidden_feature32)
        # todo 添加对应的loss输出 ！！基于video表示的余弦相似度
        past_feature,past_loss = self.past_encoder(past_traj)
        future_feature,future_loss = self.future_encoder(future_traj)
        loss_TT = past_loss + future_loss
        # ===>>>CVAE
        # qz_distribution
        # (batch_size,hidden_dim*2) 64
        h = torch.cat((past_feature, future_feature), dim=1)
        # (batch_size,32) 64->32
        h = self.out_mlp(h)
        # 在变分自编码器中，潜变量z的均值和方差是通过编码器网络输出的，并且需要满足一定的分布假设，例如高斯分布。
        # 因此，该线性层的作用是将 MLP 的输出映射到满足分布假设的潜变量均值和方差，从而使得潜变量 z 可以被正确地解码和重构。
        qz_param = self.qz_layer(h)
        qz_distribution = Normal(params=qz_param)
        qz_sampled = qz_distribution.rsample()
        # pz_distribution =》可学习的先验分析！！
        # 使用线性层对象self.pz_layer生成一个包含均值和标准差参数的向量pz_param，并以此创建一个Normal分布pz_distribution。
        pz_param = self.pz_layer(past_feature)
        pz_distribution = Normal(params=pz_param)
        # 真值
        node_past = nodes_current[:obs_length].transpose(0, 1)  # (num-ped,8,2)
        node_future = nodes_current[obs_length:].transpose(0, 1)  # (num-ped,12,2)
        # decoder
        pred_traj, recover_traj = self.decoder(past_feature, qz_sampled, node_past, sample_num=1)
        assert pred_traj.shape[0] == node_future.shape[0] == recover_traj.shape[0] == node_past.shape[0]
        batch_size = pred_traj.shape[0]
        # loss-recover,loss-pred,loss-kl
        loss_recover = self.calculate_loss_recover(recover_traj, node_past, batch_size)
        loss_pred = self.calculate_loss_pred(pred_traj, node_future, batch_size)
        batch_pednum = past_feature.shape[0]
        # todo KL-Loss分析！！min-clap 2->0
        loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_pednum, min_clip=0)
        # for loss-diversity ==》p dist for best 20 loss
        """
        主要目的在于计算loss-divers -- 由Social-GAN提出来的
        根据均值和标准差参数p_z_params（如果self.args.learn_prior为True，则使用线性层对象self.pz_layer生成），创建先验分布p(z)，用于计算ELBO损失。
        区别在于，这里对每个样本采样sample_num次，以便在计算ELBO损失时可以更精确地估计期望值。
        将past_feature张量重复sample_num次，以便将batch_size和agent_num两个维度扩展为(batch_size * agent_num * sample_num)。
        """
        # 默认20 =》 loss_diversity:基于过去的轨迹 不是结合未来和过去
        sample_num = self.args.sample_num
        # (batch_size * agent_num * sample_num,embeddings)。
        past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
        pz_params_repeat = self.pz_layer(past_feature_repeat)
        pz_distribution = Normal(params=pz_params_repeat)
        pz_sampled = pz_distribution.rsample()
        # loss-diverse
        diverse_pred_traj, _ = self.decoder(past_feature_repeat, pz_sampled, node_past, sample_num=self.args.sample_num,
                                            mode='inference')
        loss_diverse = self.calculate_loss_diverse(diverse_pred_traj, node_future)
        # total-loss!!
        total_loss = loss_pred + loss_recover + loss_kl + loss_diverse + loss_TT
        # 不同的参数组合设计 todo
        # total_loss = 1.5*loss_pred + loss_recover + loss_kl + 2.0*loss_diverse
        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item(),loss_TT.item()

    def inference(self, inputs):
        # 这是一个模型推理方法的实现。该方法接收包含过去轨迹的数据，并根据学习到的先验和给定的过去轨迹输出多样化的预测轨迹。
        # 该方法使用过去编码器计算过去特征，使用pz_layer从先验分布中采样，并使用解码器生成多样化的预测轨迹。最终输出的结果是多样性预测轨迹。
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 更新考虑从头到尾存在的行人=》数据预处理
        node_index = self.get_node_index(seq_list)
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_current, st_ed)
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:obs_length]
        # encoder
        past_feature,_ = self.past_encoder(past_traj)
        target_pred_traj = nodes_current[obs_length:]
        sample_num = self.args.sample_num
        # cvae
        past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
        pz_params_repeat = self.pz_layer(past_feature_repeat)
        pz_distribution = Normal(params=pz_params_repeat)
        pz_sampled = pz_distribution.rsample()
        node_past = nodes_current[:obs_length].transpose(0, 1)
        diverse_pred_traj, _ = self.decoder(past_feature_repeat, pz_sampled, node_past, sample_num=self.args.sample_num,
                                            mode='inference')
        #  (agent_num, sample_num, self.pred_length, 2) -> (sample_num,agent_num,self.pred_length, 2)
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
        if self.args.phase == 'test' and self.args.vis == 'sne':
            return diverse_pred_traj, target_pred_traj, past_feature
        else:
            return diverse_pred_traj, target_pred_traj
