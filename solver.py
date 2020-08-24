import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.backends import cudnn
from model.bbsnet import BBSNet
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from evaluation import fast_evaluation




np.set_printoptions(threshold=np.inf)
size = (320, 320)
size_coarse = (20, 20)


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [60,120,180]
        self.build_model()
        self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            #self.net.load_state_dict(torch.load(self.config.model))
            self.net.load_state_dict(torch.load(self.config.model))
         #为SCRN载入参数修改
        #self.net.base.load_state_dict(torch.load(r'D:\pytorch\JL-DCF\dataset\pretrained\resnet50_SCRN.pth'),strict=False)



        # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        #print(model)
        #print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net =  BBSNet()

        if self.config.cuda:
            self.net = self.net.cuda()
        # use_global_stats = True
        #self.net.apply(weights_init)
        if self.config.load != '':
            self.net.load_state_dict(torch.load(self.config.load),strict=False)  # load pretrained model
        #self.net.resnet_aspp.load_state_dict(torch.load('./pre_train/resnet34_depth.pth'))

        self.lr = self.config.lr
        self.wd = self.config.wd

        #保持caffe各层相同学习率，weight_decay

        #self.optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,self.net.parameters()), lr=self.lr, momentum=0.9)
        self.optimizer = Adam(filter(lambda p:p.requires_grad,self.net.parameters()), lr=self.lr, weight_decay=self.wd, betas=(0.99, 0.999) )
        #self.optimizer = torch.optim.Adadelta(filter(lambda p:p.requires_grad,self.net.parameters()),lr=0.01, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']),\
                                            data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device('cuda:0')
                    # cudnn.benchmark = True
                    images = images.to(device)
                    depth = depth.to(device)


                preds = self.net(images, depth)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                #flow_map = F.interpolate(flow_map, tuple(im_size), mode='bilinear', align_corners=True)
                #flow_map = np.squeeze(flow_map).cpu().data.numpy()

                pred = (pred - pred.min())/(pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                #multi_fuse = 255 * flow_map
                filename = os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        #print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        #print('Test Done!')
        print(self.config.test_fold)
        #fast_evaluation.main(self.config.test_fold, self.config.sal_mode, self.config.test_fold) #单次test
        fast_evaluation.main(self.config.test_fold, self.config.sal_mode, os.path.split(self.config.test_fold)[0]) #批量test


    # training phase
    def train(self):
        writer = SummaryWriter('train_log/run-' + time.strftime("%m-%d-")+self.config.index)
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        global step
        step = 0
        for epoch in range(self.config.epoch):
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                images, depths, gts = data_batch['images'], data_batch['depths'], data_batch['gts']

                if self.config.cuda:
                    device = torch.device('cuda:0')
                    # cudnn.benchmark = True
                    images, depths, gts = images.to(device), depths.to(device), gts.to(device)
                    #setup_seed(10)


                sal1,sal2= self.net(images, depths)
                loss1 = F.binary_cross_entropy_with_logits(sal1, gts, reduction='sum')
                loss2 = F.binary_cross_entropy_with_logits(sal2,gts,reduction='sum')
                #print(loss_r,loss_d,depth_quality)
                loss =loss1 + loss2


                

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                step += 1

                if i % 10 == 0 or i == iter_num or i == 1:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                          format(datetime.now(), epoch, self.config.epoch, i, iter_num, loss1.data, loss2.data))
                    writer.add_scalar('Loss', loss.data, global_step=step)
                    grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                    writer.add_image('RGB', grid_image, step)
                    grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                    writer.add_image('Ground_truth', grid_image, step)
                    res = sal2[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('s1', torch.tensor(res), step, dataformats='HW')
                    res = sal2[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('s2', torch.tensor(res), step, dataformats='HW')



            if (epoch + 1) % self.config.epoch_save == 0 :
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch+1))
            if epoch+1 in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(self.net.parameters(), lr=self.lr)
                #torch.save(self.net.resnet_aspp.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch+1))

        # save model
        #torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
'''
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                      epoch , self.config.epoch, i + 1, iter_num, sal_loss_fuse))
                print('Learning rate: ' + str(self.lr))
                writer.add_scalar('training loss', sal_loss_fuse ,
                                      (epoch) * len(self.train_loader.dataset) + i)   
'''
'''
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))
