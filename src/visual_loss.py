# 任务：分析loss文件 获得对应的可视化图 尽量在模型训练结束的时候进行绘画并进行存储
# 或则也可以在后期进行分析
import pandas as pd
import matplotlib.pyplot as plt
import os


# load data
def draw_loss(file_path,epoch,show=False):
    """
    file_path:文件的保存路径
    epoch:loss绘画的epoch数量
    show:False 默认不显示图片 只保存
    """
    # 读取文件并将数据存储到DataFrame中
    data = pd.read_csv(file_path, header=None, names=['epoch', 'train_loss', 'loss_pred', 'loss_recover', 'loss_kl', 'loss_diverse','loss_TT'])
    # 首先，我们需要找到最后一个从0开始的epoch的位置
    # 我们将反向遍历文件，找到第一个出现的epoch为0的行，这标志着最新的实验或运行的开始
    # 由于DataFrame已经加载，我们可以从底部向上查找epoch为0的行
    last_zero_epoch_index = data[data['epoch'] == 0].index[-1]  # 获取最后一个epoch为0的行的索引
    # 现在，我们将数据裁剪到这个索引之后的部分，这代表最新的运行数据
    # 只绘画截止对应epoch的数据
    latest_data = data.loc[last_zero_epoch_index:last_zero_epoch_index+epoch-1]
    # 重新绘制图形，展示最新的运行数据
    colors_sum = {
        'try': ['deepskyblue', 'springgreen', 'tomato', 'gold', 'orange', 'pink'],
        'good': ['#32d3eb', '#5bc49f', '#feb64d', '#ff7c7c', '#9287e7', '#60acfc']
    }
    colors = colors_sum['try']
    plt.figure(figsize=(12, 8))
    plt.plot(latest_data['epoch'], latest_data['train_loss'], label='Train Loss',color = colors[0])
    plt.plot(latest_data['epoch'], latest_data['loss_pred'], label='Prediction Loss',color = colors[1])
    plt.plot(latest_data['epoch'], latest_data['loss_recover'], label='Recovery Loss',color = colors[2])
    plt.plot(latest_data['epoch'], latest_data['loss_kl'], label='KL Loss',color = colors[3])
    plt.plot(latest_data['epoch'], latest_data['loss_diverse'], label='Diversity Loss',color = colors[4])
    plt.plot(latest_data['epoch'], latest_data['loss_TT'], label='TT loss',color=colors[5])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Latest Run: Loss Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()
    else:
        # 获取除了最后一级之外的目录路径
        dir_path = os.path.dirname(file_path)
        # 拼接新的文件名
        new_file_path = os.path.join(dir_path,str(epoch)+"_five_loss.png")
        plt.savefig(new_file_path)

if __name__ == '__main__':
    # 绘画对应的代码进行分析
    file_path = '/workspace/trajectron/STAR/output/ETH_UCY/eth/Dual_TT_origin/NoPE_MLP_DualTT_LossAlign/star_cvae_log_curve.txt'
    epoch = 200
    draw_loss(file_path=file_path, epoch=epoch,show=True)

