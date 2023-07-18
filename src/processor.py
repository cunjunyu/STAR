import torch
import torch.nn as nn

from .star import STAR
from .utils import *

from tqdm import tqdm


class processor(object):
    def __init__(self, args):

        self.args = args
        # 加载数据与模型，设置优化率
        self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args)

        self.set_optimizer()

        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()
        # 分析模型文件夹是否存在，若不存在，则创立一个
        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        # 将模型的结构写入文本文件 net.txt中，
        # 初始化变量等 'a+' 表示在文件末尾追加新内容，并允许读取文件内容。如果文件不存在，则会创建一个新文件。
        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):
        # 保存模型的代码与maml的代码框架合计
        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        test_error, test_final_error = self.test_epoch()
        print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set,
                                                                                          self.args.load_model,
                                                                                       test_error, test_final_error))
    def train(self):

        print('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):
            # todo 将模型设置为训练模式 特定条件下有效，只针对部分？？
            self.net.train()
            # train-epoch
            train_loss = self.train_epoch(epoch)
            if epoch >= self.args.start_test:
                # 如果当前轮数大于或等于 args.start_test，则将模型设置为评估模式（self.net.eval()），后续每轮都会跑
                self.net.eval()
                # todo 这里的test-epoch的数据输入与最终的test的时候是一致的，那么其相应的不就是在训练的使用了测试数据吗，
                #  此处的数据应该是train分出来的val才对
                test_error, test_final_error = self.test_epoch()
                # 调用 test_epoch 函数计算模型在测试集上的 ADE 和 FDE，并将其存储在 test_error 和 test_final_error 变量中。
                # 然后，判断当前的 FDE 是否优于历史最佳 FDE，如果是，则更新 best_ade、best_fde 和 best_epoch 变量的值，并调用 save_model 函数保存模型。
                self.best_ade = test_error if test_final_error < self.best_fde else self.best_ade
                self.best_epoch = epoch if test_final_error < self.best_fde else self.best_epoch
                self.best_fde = test_final_error if test_final_error < self.best_fde else self.best_fde
                # todo fde与ade两个指标之间的比较，best-epoch取fde？ 此处的save-model或许可以只保存效果更好的？
                self.save_model(epoch)
            # 将训练损失、ADE、FDE 和学习率等信息写入日志文件 log_file_curve 中
            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')
            # 并判断当前轮数是否为 10 的倍数，如果是，则关闭日志文件并重新打开以防止文件过大
            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')
            # 如果 start_test 大于等于当前轮数，打印当前轮数的训练损失、ADE 和 FDE，否则则只打印训练损失。
            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                        .format(epoch, train_loss, test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}'
                      .format(epoch, train_loss))

    def train_epoch(self, epoch):
        # 重置训练数据集的指针，使得数据训练从第一帧开始；
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            # todo 获取对应训练batch的数据 相应的有旋转操作以及基于观测点位置坐标的归一化操作
            #  -- 此处需要更改 -- 先观察原始的batch的格式 而后基于其更改
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            # 将数据转成pytorch的tensor格式
            inputs = tuple([torch.Tensor(i) for i in inputs])
            # 将其转移到GPU上
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            """
            分析相应的各个数据项的含义，并寻找是否可以基于此进行task的拆分，而不是从最原始的数据处理开始拆分task
            batch_abs：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标，未归一化
            batch_norm ：(seq_length, num_Peds，2) 数据经过坐标归一化后的结果
            shift_value：(seq_length, num_Peds，2) 观测序列最后一帧数据的复制
            seq_list:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
            {{注意力机制中的掩码详解：允许我们发送不同长度的批次数据一次性的发送到transformer中，在代码中是通过将所有序列填充到相同的长度，
            然后使用“atten-mask”--seq-list张量来识别那些令牌是填充的；在显存允许的情况下，使用批处理输入的速度更快，但不同的输入序列具有不同长度，
            他们无法直接组合成一个张量，此时可以用虚拟标记填充较短的序列，以使每个序列具有相同的长度；
            在注意力掩码中，输入是0和1，但在最终的计算时，会将在无效位置的注意力权重设为一个很小的值，通常为负无穷，以便在计算注意力分数时将其抑制为接近0的概率
            相应的在后续进过softmax函数时，softmax会对输入值进行指数运算，然后进行归一化，当输入值非常小或负无穷时，经过指数运算后会接近0.
            排除无效位置的影响，通过将无效位置的注意力权重设置为负无穷，可以有效的将这些位置的权重压低，}}
            nei_list：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
            nei_num：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
            batch_pednum：list 表示该batch下每个时间窗口中的行人数量
            """
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            # todo ? 整体将序列中的最后一帧的数据删除 20帧-》19帧
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum
            # 梯度清零
            self.net.zero_grad()
            # outputs (seq_length,batch-pednum,2)(eg:0-18,0-264,2) 和inputs-forward[0]的形式一致
            outputs = self.net.forward(inputs_forward, iftest=False)
            # lossmask 表示当前帧和上一帧中是否都存在数据。该掩码用于计算损失函数时去除缺失数据的贡献，避免缺失数据对损失函数的计算造成影响。
            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # todo ? 此处的损失（MSE）计算似乎同时计算了观测的值和预测的值，是整个20s序列的；不单单是未来12s。
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)

            loss += (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()
            # 损失反向传播 梯度裁剪 优化器的step函数 todo 此处后续结合meta
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)

            self.optimizer.step()

            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss.item(),
                                                                                               end - start))
        # 计算所有训练batch的平均损失
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        for batch in tqdm(range(self.dataloader.testbatchnums)):
            # 与train相同的batch处理步骤
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            all_output = []
            # 同一组数据 预测20次
            for i in range(self.args.sample_num):
                # 相应的设置iftest标记为true 测试模式
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)

            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # todo 相较于train的MSE 此处只计算在整个时间窗口都存在的行人的损失
            error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_norm[1:, :, :2],
                                                                        self.args.obs_length, lossmask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
