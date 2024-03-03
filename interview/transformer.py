import numpy as np
import torch
import torch.nn as nn
import math

# 0108 完成transformer multi-head attention机制的学习以及位置编码以及 softmax函数的编码实现
class MultiHeadAttention(nn.Module):
    def __init__(self, head, model_dim):
        super(MultiHeadAttention, self).__init__()
        # q k v out 4个linear
        self.head = head
        self.model_dim = model_dim
        self.dropout = nn.Dropout(0.1)
        self.q_w = nn.Linear(model_dim, model_dim)
        self.k_w = nn.Linear(model_dim, model_dim)
        self.v_w = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        # 输入数据维度 （B，Len,model_dim）

    def forward(self, input, mask):
        # 依据multi-head attention分析
        # 分析multi-head
        assert self.model_dim % self.head == 0
        self.head_dim = self.model_dim / self.head
        # 一般情况下query key value在此处其是是一个值
        # 变换维度
        batch_size = input.size(0)
        query = self.q_w(input).view(batch_size, -1, self.head, self.head_dim).transpose(1, 2)
        key = self.k_w(input).view(batch_size, -1, self.head, self.head_dim).transpose(1, 2)
        value = self.v_w(input).view(batch_size, -1, self.head, self.head_dim).transpose(1, 2)
        # 分析计算attention
        q_k = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 依据掩码将部分需要更改的删除
        if mask is not None:
            mask = mask.unsqueeze(1)  # 在head维度进行扩张
            q_k = q_k.masked_fill(mask == 0, float(-1e20))
        scores = nn.Softmax(q_k, -1)  # 需要明确维度
        # ===>softmax 手写一下
        scores = self.dropout(scores)
        result = torch.matmul(scores, value)
        # 维度为[B,head,len,head_dim] => 需要转回来
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.head)
        # output
        result = self.out(result)
        return result


class PostentionEncoder(nn.Module):
    def __init__(self, model_dim, concat):
        super(PostentionEncoder, self).__init__()
        self.model_dim = model_dim
        self.concat = concat
        self.fc = nn.Linear(2 * model_dim, model_dim)

    def forward(self, max_len, input):
        pe = torch.zeros(max_len, self.model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母
        div_dim = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-np.log(10000.0) * (self.model_dim)))
        pe[:, 0::2] = torch.sin(position * div_dim)
        pe[:, 1::2] = torch.cos(position * div_dim)
        # 此处的max——len即长度最大 但实际输入序列长度没有这么大 故而需要进行截取 即以0-len即可
        # 后续继续用于 和对应的特征维度进行拼接 input [batch，len，model_dim]
        seq_len = input.size(1)
        t_offset = 0
        # 截取长度
        pe = pe[t_offset:t_offset + seq_len, :]
        # 扩张 增添一个维度 并复制input.size(0)倍
        pe = pe.unsqueeze(0).repeat(input.size(0), 1, 1)
        # 两张方式
        if self.concat:
            feat = [input, pe]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x = input + pe
        return x


def softmax_ours(x, batch_size):
    """
    softmax(x) =softmax(x+c)
    由于指数函数的放大作用过于明显，如果直接使用softmax计算公式
    进行函数实现，容易导致数据溢出(上溢)。所以我们在函数实现时利用其性质：先对输入数据进行处理，之后再利用计算公式计算
    查找每个向量x的最大值c；
    每个向量减去其最大值c, 得到向量y = x-c;
    利用公式进行计算,softmax(x) = softmax(x-c) = softmax(y)
    =》输入维度为【B，Head，len，len】
    """
    # keepdims = True 使得广播
    row_max = np.max(x, axis=-1, keepdims=True)  # 保持维度
    x -= row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=-1, keepdims=True)  # 保持维度
    s = x_exp / x_sum
    return s


def softmax_ours_tensor(x, batch_size):
    # 在最后一个维度上找到最大值
    # torch.max 函数当应用于一个维度时，返回一个包含两个元素的元组：最大值以及其索引位置 故而此处运用[0]直接取max位置
    row_max = torch.max(x, dim=-1, keepdim=True)[0]  # 使用keepdim保持维度
    x = x - row_max
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp, dim=-1, keepdim=True)  # 在最后一个维度上求和
    s = x_exp / x_sum
    return s
