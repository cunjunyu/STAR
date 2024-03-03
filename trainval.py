import argparse
import ast
import os
import random
import numpy as np
import torch
import yaml
from src.processor import processor


# 设置随机种子
def set_seed(seed):
    # 之前设置的都为0
    """为多个模块设置同一随机种子以确保可重复性。"""
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch random number generator.
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python环境变量
    """
    在Python中，某些版本（3.3及以上）默认启用了哈希随机化（hash randomization），以提高字典和集合操作的安全性。
    这意味着每次启动Python解释器时，字符串和其他数据类型的哈希值会不同。设置PYTHONHASHSEED环境变量可以禁用这种哈希随机化，
    确保程序每次运行时字符串的哈希值保持不变，从而增加代码的可重复性。
    """
    # 如果你的代码使用或计划使用CUDA，则还应该设置以下内容
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个CUDA设备
    """
    当使用CUDA进行加速时，PyTorch会利用NVIDIA的cuDNN库来优化深度学习操作。
    将torch.backends.cudnn.deterministic设置为True可以使得一些cuDNN算法和操作变得确定性。
    这意味着对于相同的输入和网络配置，无论何时运行，都会得到完全相同的结果。
    """
    torch.backends.cudnn.deterministic = True
    # 禁用cuDNN的自动调优（auto-tuner）。cuDNN库可以根据当前的网络配置自动选择最优的算法来加速计算。
    # 注意：这可能会影响性能（可能会降低速度）
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser(description='STAR')
    # todo 后续需要添加对应的数据数据集迁移以及新数据的实验
    parser.add_argument('--dataset', default='ETH_UCY', type=str,
                        help='set this value to [ETH_UCY,SDD,NBA,Soccer,Ship,Ship_FVessel]')
    parser.add_argument('--SDD_if_filter', type=str, default='True', help='是否只需要行人数据集')
    parser.add_argument('--SDD_from', default='sdd_origin', type=str,
                        help='确定SDD的数据来源，sdd_origin 最原始的 sdd_exist PECNet数据')
    # 路径
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
    parser.add_argument('--save_dir', help='p.save_base_dir + str(p.dataset) +  str(p.test_set) ')
    parser.add_argument('--model_dir',help='p.save_dir + str(p.train_model) + + str(p.stage) + str(p.param_diff)')
    parser.add_argument('--config',help='参数保存路径')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--device', type=int, default=3, help='GPU选择')
    parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ,SDD,soccer] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ,SDD')

    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='star', type=str,
                        help='Your ModelStrategy name star/new_star/new_star_hin/new_star_ship/Dual_TT')
    parser.add_argument('--load_model', default=None, type=str,
                        help="load pretrained ModelStrategy for test or training, 需要的时候相应传入str，会组成model_str进行后续计算")
    parser.add_argument('--ModelStrategy', default='star.STAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int, help='注意test时，是否可以变换，以及不同的数据集也会改')
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    # 似乎没什么用？batch-size；其有batch-around-ped决定了
    parser.add_argument('--test_batch_size', default=4, type=int)

    parser.add_argument('--start_test', default=10, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)

    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    # todo 此处不同的数据集 这个是否需要进行调整 eth-ucy用的是m sdd用的是pixel!!! 需要呀！ 你之前发现了咋不去调节 无语！
    parser.add_argument('--neighbor_thred', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--inner_learning_rate', default=0.0015, type=float, help='需要考虑设置为多少，先设置和star默认的一样')
    parser.add_argument('--outer_learning_rate', default=0.0015, type=float, help='内外循环不同的学习率，需要考虑是全程一致，还是要有减少')
    parser.add_argument('--task_learning_rate', default=0.0015, type=float, help='内循环学习率')
    parser.add_argument('--clip', default=1, type=int) # todo
    parser.add_argument('--second_order', type=str, default="False", help='Dropout_rate_value')
    parser.add_argument('--first_order_to_second_order_epoch', type=int, default=-1, help='maml改进')
    # Meta
    parser.add_argument('--batch_around_ped', default=256, type=int, help='一个batch所应该包含的行人数，可以调节一下分析')
    parser.add_argument('--batch_around_ped_meta', default=256, type=int, help='meta一个batch所应该包含的行人数，可以调节一下分析')
    parser.add_argument('--meta_way', type=str, default='sequential1', help='可以用两种选项，即parallel2并行和sequential1串行')
    parser.add_argument('--query_sample_num', default=4, type=int, help='每个数据集中的support采集对应几个query')
    parser.add_argument('--stage', default='origin', type=str, help='决定是否进行meta训练 可选origin MLDG MVDG MVDGMLDG')
    parser.add_argument('--optim_trajectory_num', default=3, type=int, help='优化轨迹数量')
    # CVAE
    parser.add_argument('--ztype', default='gaussian', type=str, help='选择创建哪种分布类型的后验分布q(z|x,y)')
    parser.add_argument('--zdim', default=16, type=int, help='对应的z均值和方差的维度')
    parser.add_argument('--min_clip', type=float, default=2.0) # KL散度设置
    parser.add_argument('--learn_prior', action='store_true', default=False) # 学习先验分布
    # mixup
    parser.add_argument('--lam', type=float, default=0.05, help='在元测试时注入特征')
    parser.add_argument('--ifmixup', default=False, type=ast.literal_eval, help='确定是否运用混合特征注入')
    # HIN
    parser.add_argument('--relation_num', default=3, type=int, help='确定相应的不同数据集内agent类型的数量 ')
    parser.add_argument('--HIN', default='False', type=str, help='确定是否需要运用HIN结构，以及相应的数据处理')
    # 可视化分析
    parser.add_argument('--vis', type=str, default='None', help='分析是否需要以及何种可视化None：无，sne，traj, traj_comparison')
    parser.add_argument('--k_best', type=int, default=1, help='分析SDD可视化中需要的误差数量')
    parser.add_argument('--denormalize', type=str, default='True', help='确定是否需要归一化')
    # 新模型 Dual_TT
    parser.add_argument('--PE', type=str, default='False', help='是否需要进行位置编码 True/False')
    parser.add_argument('--param_diff', type=str, default='origin', help='用于标记不同的参数组合从而确保运行多个代码但互相不影响')
    parser.add_argument('--Dual_TT_ablation', type=str, default='Dual_TT', help='模型的消融实验：Dual_TT,IT,TI,II,TT')
    # parser.add_argument('--time_embedding',default=True,type=ast.literal_eval,help='确定如何从多个维度转变为一个')
    return parser


def load_arg(p):
    # save arg
    # 判断配置文件是否存在。如果指定的配置文件 p.config 存在，则打开文件，读取其中的配置参数，并将它们存储在 default_arg 字典中
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # 使用了 PyYAML 库的 load 方法来加载 YAML 格式的配置文件。由于 PyYAML 5.1 之后的版本中默认禁止加载 Python 对象（因为存在安全漏洞）
            # 因此需要使用 Loader=yaml.FullLoader 参数来指定使用完整的 YAML 解析器。
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        # 检查配置参数是否正确。将读取到的配置参数 default_arg 中的键值对与命令行参数中的键值对进行比较，
        # 如果有某个参数在 default_arg 中但不在命令行参数中，则提示错误信息，并断言该参数必须存在于命令行参数中
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        # parser.set_defaults(**default_arg) 将 default_arg 中定义的参数作为默认参数添加到解析器中。
        # 这意味着如果命令行参数没有显式地指定这些参数，它们将使用默认值。
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    # 将命令行参数转换为字典
    arg_dict = vars(args)
    # 创建目录。如果指定的保存模型的目录 args.model_dir 不存在，则使用 os.makedirs() 函数创建目录。
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    #  写入配置文件。使用 Python 的内置模块 yaml 的 dump() 方法，将字典 arg_dict 中的键值对写入到配置文件 args.config 中。
    #  这个过程中，yaml 模块会自动将字典中的数据类型转换为 YAML 格式。YAML 更加易读易写，可以直观地表示复杂的数据结构。
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    # 开启异常检测 检测完成 找到错误后 可关闭
    # torch.autograd.set_detect_anomaly(True)
    set_seed(0)
    parser = get_parser()
    p = parser.parse_args()
    # test-set 可以设置为SDDNBA,NFL等其他数据集
    # save_base_dir() -> save_dir(测试数据集) ->  data_dir(相同数据+相同模型) -> model_dir(相同数据+相同模型+不同参数)
    p.save_dir = p.save_base_dir + str(p.dataset) + '/' + str(p.test_set) + '/'
    p.data_dir = p.save_base_dir + str(p.dataset) + '/' + str(p.test_set) + '/' + str(p.train_model) + '_' + str(
        p.stage) + '/'
    p.model_dir = p.save_base_dir + str(p.dataset) + '/' + str(p.test_set) + '/' + str(p.train_model) + '_' + str(
        p.stage) + '/' + str(p.param_diff) + '/'

    p.config = p.model_dir + 'config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)
    # 首次运行文件时，无配置的yaml文件则会运行save-arg保存默认参数，而后续的实现则是命令行参数和默认参数的结合分析
    args = load_arg(p)
    device = args.device
    torch.cuda.set_device(device)

    trainer = processor(args)

    if args.phase == 'test' and args.vis != 'None':
        trainer.test_vis()
    elif args.phase == 'test' and args.vis == 'None':
        trainer.test()
    elif args.phase == 'train':
        trainer.train()
    else:
        raise NotImplementedError

    # 关闭异常检测 找到错误后进行关闭
    # torch.autograd.set_detect_anomaly(False)
