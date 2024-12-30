import torch
import torch.nn as nn

from .star import STAR
from .utils import *

from tqdm import tqdm


class processor(object):
    def __init__(self, args):

        self.args = args

        self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args)

        self.set_optimizer()

        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file_name = "net_" + self.args.train_model + "_" + self.args.encoder_type + "_" + str(args.num_epochs) + ".txt"

        self.log_file_name = "log_curve_" + self.args.train_model + "_" + self.args.encoder_type + "_" + str(args.num_epochs) + ".txt"


        self.net_file = open(os.path.join(self.args.model_dir, self.net_file_name), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, self.log_file_name), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + self.args.encoder_type + '.tar'
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

            self.net.train()
            train_loss = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error = self.test_epoch()
                self.best_ade = test_error if test_final_error < self.best_fde else self.best_ade
                self.best_epoch = epoch if test_final_error < self.best_fde else self.best_epoch
                self.best_fde = test_final_error if test_final_error < self.best_fde else self.best_fde
                self.save_model(epoch)

            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')

            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, self.log_file_name), 'a+')

            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                        .format(epoch, train_loss, test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}'
                      .format(epoch, train_loss))

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)

            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)

            loss += (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()

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

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            all_output = []
            for i in range(self.args.sample_num):
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)

            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_norm[1:, :, :2],
                                                                        self.args.obs_length, lossmask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
