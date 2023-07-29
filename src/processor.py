import time

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from .star import STAR
from .utils import *

from tqdm import tqdm
import copy


class processor(object):
    def __init__(self, args):

        self.args = args
        self.device = args.device
        # 加载数据与模型，设置优化率
        self.dataloader = Trajectory_Dataloader(args)
        # 设置两个不同的模型 模型结构一致 参数不一致 相应的net用于外循环 trajectory用作内循环 每次传递参数
        self.net = STAR(args)
        # todo 对于学习率优化方面 需要添加对应的内外循环各自的代码
        self.set_optimizer()
        self.task_learning_rate = self.args.task_learning_rate

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
        self.load_model()
        self.net.train()
        test_error, test_final_error = 0, 0
        if self.args.load_model is  not None:
            epoch_start = int(self.args.load_model)
        else:
            epoch_start = 0
        for epoch in range(epoch_start,self.args.num_epochs):

            train_loss = self.train_meta_epoch(epoch)
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
            # if epoch % 10 == 0:
            if epoch % 2 == 0:
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
            # 整体将序列中的最后一帧的数据删除 20帧-》19帧
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum
            # 梯度清零
            self.net.zero_grad()
            # outputs (seq_length,batch-pednum,2)(eg:0-18,0-264,2) 和inputs-forward[0]的形式一致
            names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            weights_dict = self.net.state_dict()

            # print(names_weights_copy)
            outputs = self.net.forward(inputs_forward, iftest=False)
            # lossmask 表示当前帧和上一帧中是否都存在数据。该掩码用于计算损失函数时去除缺失数据的贡献，避免缺失数据对损失函数的计算造成影响。
            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # 此处的损失（MSE）计算似乎同时计算观测的值和预测的值，是整个20s序列的；不单单是未来12s。
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)

            loss += (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()
            # 损失反向传播 梯度裁剪 优化器的step函数
            loss.backward()
            """
            对神经网络模型的梯度进行裁剪;self.net.parameters() 表示获取神经网络模型中所有可训练参数的迭代器。
            然后，nn.utils.clip_grad_norm_() 函数会计算出所有参数的梯度的范数并进行裁剪，以避免梯度爆炸的问题。
            self.args.clip 是梯度裁剪的阈值，表示梯度的范数超过这个值时，会将梯度进行缩放，以使梯度的范数不超过这个阈值。
            clip_grad_norm_() 函数的返回值是裁剪后的梯度范数值;这个函数会修改模型参数的梯度值，
            因此在调用 backward() 函数计算梯度之后，但在执行 step() 函数更新参数之前，通常会使用这个函数对梯度进行裁剪。
            """
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

    def train_meta_epoch(self,epoch):
        """
        结合Meta框架重写该部分代码；
        内外循环
        """
        # 第一步依据完整数据拆分出batch list
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        for batch_task_id,batch_task_data in enumerate(self.dataloader.train_batch_task):
            # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
            #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
            # 针对batch-task-data中的4个task进行处理 list
            print('begin' +str(epoch)+str(batch_task_id))
            start = time.time()
            self.net.zero_grad()
            # !!!!(1)注意的是 state——dict是浅拷贝，即net-initial-dict改变的话，那么当你修改param，相应地也会修改model的参数。
            # model这个对象实际上是指向各个参数矩阵的，而浅拷贝只会拷贝最外层的这些“指针；
            # from copy import deepcopy  best_state = copy.deepcopy(model.state_dict()) 深拷贝 互不影响
            net_initial_dict = copy.deepcopy(self.net.state_dict())
            task_query_loss = []
            for task_id,task_batch_data in enumerate(batch_task_data):
                # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
                print('begin'+str(epoch)+'--'+str(batch_task_id)+'---'+str(task_id))
                # 1 !!!! (2)每次都从1开始 清零 初始化一个self.new——model的话会导致反复更新累加 导致原位操作 7-9
                new_model = STAR(self.args).cuda()
                # 2
                new_model.load_state_dict(net_initial_dict)
                new_model.zero_grad()
                # 准备数据
                support_set_inital = task_batch_data[1]
                query_set_inital = task_batch_data[0]
                # todo forward需要与对应的参数结合起来  内外参数
                support_loss = self.meta_forward(new_model, support_set_inital, stage='support')
                # 计算grad
                names_weights_copy = self.get_inner_loop_parameter_dict(new_model.named_parameters())
                # create_graph,retain_graph的取值
                grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=False,retain_graph=False, allow_unused=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key:names_weights_copy[key]-self.task_learning_rate*inner_dict_grads[key] for key in names_weights_copy.keys()}
                # 3加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
                new_model.load_state_dict(new_inner_dict)
                # 按理此处没有grad？
                new_model.zero_grad()
                del grads
                query_loss = self.meta_forward(new_model, query_set_inital, stage='query')
                task_query_loss.append(query_loss)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            print('task_query_loss:'+str(task_query_loss.cpu().detach().numpy()))
            loss_epoch =loss_epoch + task_query_loss.item()
            """
            # ！！！（3）
            todo task_query_loss是由内部的new_model计算得到的，loss backward只会计算new_model网络的梯度，此时其初始值是不同于self.net的
            我们后期只需要他的梯度，不需要他的值，故而设计函数将梯度对应传回来即可
            torch1.5以下，不会监查原位操作的问题，但相应的其实其梯度计算错误。
            """
            task_query_loss.backward()
            for old,new in zip(self.net.named_parameters(),new_model.named_parameters()):
                # 返回一个tuple 【名称，参数tensor】
                old[1].grad = new[1].grad
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                        batch_task_id, len(self.dataloader.train_batch_task), epoch,task_query_loss.item(),end - start))
        train_loss_epoch = loss_epoch/len(self.dataloader.train_batch_task)
        return train_loss_epoch

    def meta_forward(self,model,data,stage,optim=None):
        """
        loss 不在这里面求解
        """
        model.train()
        if optim is not None:
            optim.zero_grad()
        data_set = self.dataloader.rotate_shift_batch(data[0])
        # 将数据转成pytorch的tensor格式 将其转移到GPU上
        data_set = tuple([torch.Tensor(i) for i in data_set])
        data_set = tuple([i.cuda() for i in data_set])
        loss = torch.zeros(1).cuda()
        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = data_set
        set_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum
        # todo forward需要与对应的参数结合起来  内外参数
        print('begin '+stage)
        # todo iftest ?
        outputs = model.forward(set_forward, iftest=False)
        lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:],using_cuda=self.args.using_cuda)
        loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
        loss = loss + torch.sum(loss_o * lossmask / num)
        # loss.backward(create_graph=create_graph,retain_graph=True)
        if optim is not None:
            optim.step()
        return loss

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        将相应的参数转移到指定设备上，并排除部分不需要优化的参数，或特殊要求下的参数；最终获得参数的字典，需要关注后续如何将参数字典给到模型
        """
        return {
            name: param.to(device=self.device)
            for name, param in params
            if param.requires_grad
        }

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

    def train_meta_epoch_false(self, epoch):
        """
        结合Meta框架重写该部分代码；
        内外循环
        """
        # 第一步依据完整数据拆分出batch list
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_task):
            # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
            #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
            # 针对batch-task-data中的4个task进行处理 list
            print('begin' + str(batch_task_id))
            start = time.time()
            self.net.zero_grad()
            # net_initial_dict 包含值 device
            net_initial_dict = self.net.state_dict()
            task_query_loss = []
            for task_id, task_batch_data in enumerate(batch_task_data):
                # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
                print('begin' + str(batch_task_id) + '---' + str(task_id))
                self.trajectory_pred.load_state_dict(net_initial_dict)
                self.trajectory_pred.zero_grad()
                # 准备数据
                support_set_inital = task_batch_data[0]
                query_set_inital = task_batch_data[1]
                support_loss = self.meta_forward(self.trajectory_pred, support_set_inital, stage='support')
                # 依据support loss更新参数
                # inner_dict = self.get_inner_loop_parameter_dict(inner_dict)
                names_weights_copy = self.get_inner_loop_parameter_dict(self.trajectory_pred.named_parameters())
                # names_weights_copy 包含值 device 还有对应的requires_grad
                grads = torch.autograd.grad(support_loss, names_weights_copy.values(),create_graph=False,allow_unused=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                                  in names_weights_copy.keys()}
                # 加载内循环更新完的参数 以新参数计算query的loss
                self.trajectory_pred.load_state_dict(new_inner_dict)
                self.trajectory_pred.zero_grad()
                query_loss = self.meta_forward(self.trajectory_pred, query_set_inital, stage='query')
                task_query_loss.append(query_loss)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
            loss_epoch = loss_epoch + task_query_loss.item()
            """
            # todo task_query_loss是由内部的trajectory计算得到的，但是我们此处需要用其去更新外围的参数，
            此处直接的loss backward会使得其同时计算对于trajectory和net的网络的梯度，故而相应的使用已经更新过的参数去更新参数，
            会导致原位操作的问题；而相应的分析，此处我们只需要更新self.net的参数，故而可以进行指定（torch1.8以上可以）
            torch1.5以下，不会监查原位操作的问题，但相应的其实其梯度计算错误。
            ==>> 事实是一直没更新，backward与step未正确的更新参数 
            """
            task_query_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                        batch_task_id, len(self.dataloader.train_batch_task), epoch, task_query_loss.item(),
                        end - start))
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_task)
        return train_loss_epoch

    """
    舍弃错误代码
    def train_meta_epoch_v4(self,epoch,first_order=False):
        # 第一步依据完整数据拆分出batch list
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        for batch_task_id,batch_task_data in enumerate(self.dataloader.train_batch_task):
            # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
            #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
            # 针对batch-task-data中的4个task进行处理 list
            print('begin' + str(batch_task_id))
            start = time.time()
            self.net.zero_grad()
            task_query_loss = []
            for task_id,task_batch_data in enumerate(batch_task_data):
                # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
                print('begin'+str(batch_task_id)+'---'+str(task_id))
                new_model = STAR(self.args).cuda()
                # todo debug
                new_model.copy(self.net, same_var=True)
                # 准备数据
                query_set_inital = task_batch_data[0]
                support_set_inital = task_batch_data[1]
                support_loss = self.meta_forward_v2(new_model, support_set_inital, stage='support')
                # 计算第一次的梯度
                new_model.zero_grad()
                names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
                grads = torch.autograd.grad(support_loss,names_weights_copy,create_graph=True,retain_graph=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                                  in names_weights_copy.keys()}
                # todo 加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
                for name,param in new_model.named_parameters():
                    if name in new_inner_dict:
                        new_model.set_param(new_model,name,new_inner_dict[name])
                    else:
                        new_model.set_param(new_model,name,param)
                del grads
                query_loss = self.meta_forward_v2(new_model,query_set_inital,stage='query')
                query_loss.backward(create_graph=False,retain_graph=True)
                task_query_loss.append(query_loss)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            print('task_query_loss:'+str(task_query_loss.cpu().detach().numpy()))
            loss_epoch =loss_epoch + task_query_loss.item()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # new_model.zero_grad()
            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                        batch_task_id, len(self.dataloader.train_batch_task), epoch,task_query_loss.item(),end - start))
        train_loss_epoch = loss_epoch/len(self.dataloader.train_batch_task)
        return train_loss_epoch
        
    def train_meta_epoch_v3(self,epoch,first_order=False):
        # 第一步依据完整数据拆分出batch list 结合Meta框架重写该部分代码；
        #         内外循环
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        for batch_task_id,batch_task_data in enumerate(self.dataloader.train_batch_task):
            # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
            #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
            # 针对batch-task-data中的4个task进行处理 list
            print('begin' + str(batch_task_id))
            start = time.time()
            self.net.zero_grad()
            task_query_loss = []
            for task_id,task_batch_data in enumerate(batch_task_data):
                # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
                print('begin'+str(batch_task_id)+'---'+str(task_id))
                new_model = STAR(self.args).cuda()
                new_model.copy(self.net,same_var=True)
                # 准备数据
                query_set_inital = task_batch_data[0]
                support_set_inital = task_batch_data[1]
                support_loss = self.meta_forward(new_model, support_set_inital, stage='support', create_graph=not first_order)
                for name,param in new_model.named_params():
                    grad = param.grad
                    if first_order:
                        grad = V(grad.detach().data)
                    new_model.set_params(name,param-self.task_learning_rate*grad)

                    # new_model.zero_grad()
                query_loss = self.meta_forward(new_model,query_set_inital,stage='query',create_graph=False)
                task_query_loss.append(query_loss)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            print('task_query_loss:'+str(task_query_loss.cpu().detach().numpy()))
            loss_epoch =loss_epoch + task_query_loss.item()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # new_model.zero_grad()
            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                        batch_task_id, len(self.dataloader.train_batch_task), epoch,task_query_loss.item(),end - start))
        train_loss_epoch = loss_epoch/len(self.dataloader.train_batch_task)
        return train_loss_epoch
        
    def meta_forward_v1(self,model,data,stage,create_graph=False,optim=None):
        model.train()
        if optim is not None:
            optim.zero_grad()
        data_set = self.dataloader.rotate_shift_batch(data[0])
        # 将数据转成pytorch的tensor格式 将其转移到GPU上
        data_set = tuple([torch.Tensor(i) for i in data_set])
        data_set = tuple([i.cuda() for i in data_set])
        loss = torch.zeros(1).cuda()
        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = data_set
        set_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum
        # todo forward需要与对应的参数结合起来  内外参数
        print('begin '+stage)
        # todo iftest ?
        outputs = model.forward(set_forward, iftest=False)
        lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:],using_cuda=self.args.using_cuda)
        loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
        loss = loss + torch.sum(loss_o * lossmask / num)
        loss.backward(create_graph=create_graph,retain_graph=True)
        if optim is not None:
            optim.step()
        return loss
        
    def train_meta_epoch_v2(self,epoch):
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch =[]
        for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_task):
            print('begin' + str(batch_task_id))
            start = time.time()
            self.net.zero_grad()
            net_initial_dict = copy.deepcopy(self.net.state_dict())
            task_query_loss =[]
            for task_id, task_batch_data in enumerate(batch_task_data):
                print('begin' + str(batch_task_id) + '---' + str(task_id))
                self.net.load_state_dict(net_initial_dict)
                # 准备数据 support 不全
                # support_set_inital = task_batch_data[0]
                support_set_inital = task_batch_data[1]
                # 数据旋转 基于观测帧归一化 只取batch-data 忽略 batch-id
                support_set = self.dataloader.rotate_shift_batch(support_set_inital[0])
                # 将数据转成pytorch的tensor格式 将其转移到GPU上
                support_set = tuple([torch.Tensor(i) for i in support_set])
                support_set = tuple([i.cuda() for i in support_set])
                # 准备数据  query 有完整的格式
                query_set_inital = task_batch_data[0]
                query_set = self.dataloader.rotate_shift_batch(query_set_inital[0])
                # 将数据转成pytorch的tensor格式 将其转移到GPU上
                query_set = tuple([torch.Tensor(i) for i in query_set])
                query_set = tuple([i.cuda() for i in query_set])
                support_loss = torch.zeros(1).cuda()
                support_batch_abs, support_batch_norm, support_shift_value, support_seq_list, support_nei_list, \
                support_nei_num, support_batch_pednum = support_set
                support_set_forward = support_batch_abs[:-1], support_batch_norm[:-1], support_shift_value[
                                                                                       :-1], support_seq_list[:-1], \
                                      support_nei_list[:-1], support_nei_num[:-1], support_batch_pednum
                # todo forward需要与对应的参数结合起来  内外参数
                print('begin support')
                # todo 原来的模型参数是直接赋值params的，那么相应的赋值不会改变其对应的net本身的参数结构 内循环情况下version不变 故而无inplace operation
                support_outputs = self.net.forward(support_set_forward, iftest=False)
                support_lossmask, num = getLossMask(support_outputs, support_seq_list[0], support_seq_list[1:],
                                                    using_cuda=self.args.using_cuda)
                support_loss_o = torch.sum(self.criterion(support_outputs, support_batch_norm[1:, :, :2]), dim=2)
                support_loss = support_loss + torch.sum(support_loss_o * support_lossmask / num)
                # 依据support loss更新参数
                # inner_dict = self.get_inner_loop_parameter_dict(inner_dict)
                names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
                # names_weights_copy 包含值 device 还有对应的requires_grad
                grads = torch.autograd.grad(support_loss, names_weights_copy.values(),
                                            create_graph=self.args.second_order and
                                                         epoch > self.args.first_order_to_second_order_epoch,
                                            allow_unused=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                                  in names_weights_copy.keys()}
                # 加载内循环更新完的参数 以新参数计算query的loss
                self.net.load_state_dict(new_inner_dict)
                print('begin query')
                query_loss = torch.zeros(1).cuda()
                query_batch_abs, query_batch_norm, query_shift_value, query_seq_list, query_nei_list, \
                query_nei_num, query_batch_pednum = query_set
                query_set_forward = query_batch_abs[:-1], query_batch_norm[:-1], query_shift_value[:-1], query_seq_list[
                                                                                                         :-1], \
                                    query_nei_list[:-1], query_nei_num[:-1], query_batch_pednum
                query_outputs = self.net.forward(query_set_forward, iftest=False)
                query_lossmask, num = getLossMask(query_outputs, query_seq_list[0], query_seq_list[1:],
                                                  using_cuda=self.args.using_cuda)
                query_loss_o = torch.sum(self.criterion(query_outputs, query_batch_norm[1:, :, :2]), dim=2)
                query_loss = query_loss + torch.sum(query_loss_o * query_lossmask / num)
                task_query_loss.append(query_loss)
                # 四次的初始参数一致，需要的是各自更新计算后的loss，参数可以摒弃
                self.net.zero_grad()
                #
                self.net.load_state_dict(net_initial_dict)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
            loss_epoch.append(task_query_loss)
            self.net.zero_grad()
            origin_name_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            # 原位操作问题 ！！
            task_query_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            test_name_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_task_id, len(self.dataloader.train_batch_task), epoch, task_query_loss.item(), end - start))
        train_loss_epoch = torch.mean(torch.stack(loss_epoch))
        return train_loss_epoch
    
    """

