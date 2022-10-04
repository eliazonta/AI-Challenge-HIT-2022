""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--batchsize', type=int, default=24, help='input batch size')
        self.parser.add_argument('--isize', type=int, default=64, help='input ES size.')
        self.parser.add_argument('--lsize', type=int, default=15, help='latent space size.')
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')

        ##
        # Train
        self.parser.add_argument('--phase', type=str, default='train', help='train, validation, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=250, help='number of epochs to train for')
        self.parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate of Generator for adam')
        self.isTrain = False
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test


        args = vars(self.opt)

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.dataset, self.opt.model)
        else:
            self.opt.name = "%s/%s" % (self.opt.dataset, self.opt.name)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
