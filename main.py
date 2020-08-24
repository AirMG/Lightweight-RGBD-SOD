import argparse
import os
from dataset.dataset import get_loader
from solver import Solver
import time


def get_test_info(sal_mode='NJU2K'):
    if sal_mode == 'NJU2K':
        image_root = 'D:/pytorch/JL-DCF/data/NJU2K_test/'
        image_source = 'D:/pytorch/JL-DCF/data/NJU2K_test/test.lst'
    elif sal_mode == 'STERE':
        image_root = 'D:/pytorch/JL-DCF/data/STERE/'
        image_source = 'D:/pytorch/JL-DCF/data/STERE/test.lst'
    elif sal_mode == 'RGBD135':
        image_root = 'D:/pytorch/JL-DCF/data/RGBD135/'
        image_source = 'D:/pytorch/JL-DCF/data/RGBD135/test.lst'
    elif sal_mode == 'LFSD':
        image_root = 'D:/pytorch/JL-DCF/data/LFSD/'
        image_source = 'D:/pytorch/JL-DCF/data/LFSD/test.lst'
    elif sal_mode == 'NLPR':
        image_root = 'D:/pytorch/JL-DCF/data/NLPR/'
        image_source = 'D:/pytorch/JL-DCF/data/NLPR/test.lst'
    elif sal_mode == 'SIP':
        image_root = 'D:/pytorch/JL-DCF/data/SIP/'
        image_source = 'D:/pytorch/JL-DCF/data/SIP/test.lst'

    return image_root, image_source

def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        """
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        """
        while os.path.exists("%s/demo-%s-%s-%d" % (config.save_folder, time.strftime("%m"), time.strftime("%d"), run)):
            run += 1
        os.mkdir("%s/demo-%s-%s-%d" % (config.save_folder, time.strftime("%m"), time.strftime("%d"), run))
        config.save_folder = "%s/demo-%s-%s-%d" % (config.save_folder, time.strftime("%m"), time.strftime("%d"), run)
        config.index = str(run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.makedirs(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = 'D:/pytorch/JL-DCF/dataset/pretrained/resnet101.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=10) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--train_list', type=str, default='')

    # Testing settings
    parser.add_argument('--model', type=str, default=None) # Snapshot
    parser.add_argument('--test_fold', type=str, default=None) # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='STERE') # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)
