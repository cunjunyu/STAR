import argparse
import ast
import os

import torch
import yaml

from src.processor import processor

# Use Deterministic mode and set random seed
# 在 cuDNN 中打开确定性模式可以确保每次运行代码时执行相同的计算？
torch.backends.cudnn.deterministic = True
# 将 torch.backends.cudnn.benchmark = False 设置为 False 可以禁用 cuDNN 的自动调整器，这可以提高计算的确定性，但可能会导致性能较慢。
torch.backends.cudnn.benchmark = False
# 将随机种子设置为固定值确保每次运行代码时都会生成相同的随机数序列，这对于结果的可重复性非常有用。
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='STAR')
    # 后续可以改成SDD NBA
    parser.add_argument('--dataset', default='eth5')
    # 需要添加新的名字，使得meta与原始的有所区别
    parser.add_argument('--save_dir',help='?')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='star', help='Your model name')
    parser.add_argument('--load_model', default=None, type=str, help="load pretrained model for test or training, 需要的时候相应传入str，会组成model_str进行后续计算")
    parser.add_argument('--model', default='star.STAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int,help='注意test时，是否可以变换，以及不同的数据集也会改')
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int,help='一个batch所应该包含的行人数，可以调节一下分析')
    parser.add_argument('--batch_size', default=8, type=int)
    # 似乎没什么用？batch-size；其有batch-around-ped决定了
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--start_test', default=10, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument('--second_order', type=str, default="False", help='Dropout_rate_value')
    parser.add_argument('--first_order_to_second_order_epoch', type=int, default=-1, help='maml改进')
    parser.add_argument('--task_learning_rate', default=0.0015, type=float, help='内循环学习率')
    parser.add_argument('--device', type=int, default=3, help='GPU选择')
    parser.add_argument('--batch_around_ped_meta', default=256, type=int, help='meta一个batch所应该包含的行人数，可以调节一下分析')
    parser.add_argument('--query_sample_num', default=4, type=int,help='每个数据集中的support采集对应几个query')
    return parser


def load_arg(p):
    # save arg
    # 判断配置文件是否存在。如果指定的配置文件 p.config 存在，则打开文件，读取其中的配置参数，并将它们存储在 default_arg 字典中
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # 使用了 PyYAML 库的 load 方法来加载 YAML 格式的配置文件。由于 PyYAML 5.1 之后的版本中默认禁止加载 Python 对象（因为存在安全漏洞）
            # 因此需要使用 Loader=yaml.FullLoader 参数来指定使用完整的 YAML 解析器。
            default_arg = yaml.load(f,Loader = yaml.FullLoader)
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
    #torch.autograd.set_detect_anomaly(True)
    parser = get_parser()
    p = parser.parse_args()

    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)
    # 首次运行文件时，无配置的yaml文件则会运行save-arg保存默认参数，而后续的实现则是命令行参数和默认参数的结合分析
    args = load_arg(p)
    device = args.device
    torch.cuda.set_device(device)

    trainer = processor(args)

    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()

    # 关闭异常检测 找到错误后进行关闭
    # torch.autograd.set_detect_anomaly(False)