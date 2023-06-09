###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import os
import torch
from modeling.sync_batchnorm.replicate import patch_replication_callback
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data import DataLoader
import modeling.dff_encoding.utils as utils
from utils.DFF_losses import EdgeDetectionReweightedLosses
from modeling.DFF import DFF
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from my_options.DFF_options import Options
from tqdm import tqdm
import scipy.io as sio

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        print(self.saver.experiment_dir)
        self.output_dir = os.path.join(self.saver.experiment_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Define Dataloader
        self.train_dataset = Mydataset(root_path=self.args.data_path, split='trainval', crop_size=self.args.crop_size)
        self.test_dataset = Mydataset(root_path=self.args.data_path, split='test', crop_size=self.args.crop_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                      num_workers=args.workers)

        self.class_num = 4
        # Define network
        model = DFF(self.class_num, backbone=self.args.backbone)

        # optimizer using different LR
        if args.model == 'dff': # dff
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},
                           {'params': model.ada_learner.parameters(), 'lr': args.lr*10},
                           {'params': model.side1.parameters(), 'lr': args.lr*10},
                           {'params': model.side2.parameters(), 'lr': args.lr*10},
                           {'params': model.side3.parameters(), 'lr': args.lr*10},
                           {'params': model.side5.parameters(), 'lr': args.lr*10},
                           {'params': model.side5_w.parameters(), 'lr': args.lr*10}]
        else: # casenet
            assert args.model == 'casenet'
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},
                           {'params': model.side1.parameters(), 'lr': args.lr*10},
                           {'params': model.side2.parameters(), 'lr': args.lr*10},
                           {'params': model.side3.parameters(), 'lr': args.lr*10},
                           {'params': model.side5.parameters(), 'lr': args.lr*10},
                           {'params': model.fuse.parameters(), 'lr': args.lr*10}]

        optimizer = torch.optim.SGD(params_list,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        self.criterion = EdgeDetectionReweightedLosses()
        self.model, self.optimizer = model, optimizer

        # Using cuda
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # finetune from a trained model
        if args.ft:
            args.start_epoch = 0
            checkpoint = torch.load(args.ft_resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
        
        # resuming checkpoint
        self.best_pred = 0.0
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader), lr_step=args.lr_step)

    def training(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        train_loss = 0.
        train_loss_all = 0.
        
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()
            image = image.cuda()
            target = target.cuda()
            target = target[:, 1:5, :, :]

            outputs = self.model(image.float())

            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_all += loss.item()
            tbar.set_description('train-loss: %.4f' % (train_loss_all / (i + 1)))

            #if i == 0 or (i+1) % 20 == 0:
                #train_loss = train_loss / min(20, i + 1)
            #    train_loss = 0.

        print('-> Epoch [%d], Train epoch loss: %.3f' % (
                         epoch + 1, train_loss_all / (i + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            if (epoch + 1) % 10 == 0:
                is_best = False
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)


    def test(self, epoch):
        print('Test epoch: %d' % epoch)
        self.output_dir = os.path.join(self.saver.experiment_dir, 'side_outs', str(epoch + 1))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.depth_output_dir = os.path.join(self.saver.experiment_dir, 'side_outs', str(epoch + 1), 'depth/mat')
        if not os.path.exists(self.depth_output_dir):
            os.makedirs(self.depth_output_dir)
        self.normal_output_dir = os.path.join(self.saver.experiment_dir, 'side_outs', str(epoch + 1), 'normal/mat')
        if not os.path.exists(self.normal_output_dir):
            os.makedirs(self.normal_output_dir)
        self.reflectance_output_dir = os.path.join(self.saver.experiment_dir, 'side_outs', str(epoch + 1), 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir):
            os.makedirs(self.reflectance_output_dir)
        self.illumination_output_dir = os.path.join(self.saver.experiment_dir, 'side_outs', str(epoch + 1), 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir):
            os.makedirs(self.illumination_output_dir)

        self.output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs', str(epoch + 1))
        if not os.path.exists(self.output_dir2):
            os.makedirs(self.output_dir2)
        self.depth_output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs', str(epoch + 1), 'depth/mat')
        if not os.path.exists(self.depth_output_dir2):
            os.makedirs(self.depth_output_dir2)
        self.normal_output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs', str(epoch + 1), 'normal/mat')
        if not os.path.exists(self.normal_output_dir2):
            os.makedirs(self.normal_output_dir2)
        self.reflectance_output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs', str(epoch + 1), 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir2):
            os.makedirs(self.reflectance_output_dir2)
        self.illumination_output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs', str(epoch + 1), 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir2):
            os.makedirs(self.illumination_output_dir2)

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            if self.args.cuda:
                image = image.cuda()
                crop_h,crop_w = image.size(2),image.size(3)
                image = image[:,:,0:crop_h-1,0:crop_w-1]
            with torch.no_grad():
                output_list = self.model(image)
                
            pred2 = output_list[0]
            pred2 = torch.sigmoid(pred2)

            pred = torch.zeros(1,4,crop_h,crop_w)
            pred[:,:,0:crop_h-1,0:crop_w-1] = pred2

            pred = pred.squeeze()
            out_depth = pred[0, :, :]
            out_normal = pred[1, :, :]
            out_reflectance = pred[2, :, :]
            out_illumination = pred[3, :, :]

            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(self.depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(self.normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(self.reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(self.illumination_output_dir, '{}.mat'.format(name)),
                        {'result': illumination_pred})

            pred2 = output_list[1]
            pred2 = torch.sigmoid(pred2)

            pred = torch.zeros(1, 4, crop_h, crop_w)
            pred[:, :, 0:crop_h - 1, 0:crop_w - 1] = pred2

            pred = pred.squeeze()
            out_depth = pred[0, :, :]
            out_normal = pred[1, :, :]
            out_reflectance = pred[2, :, :]
            out_illumination = pred[3, :, :]

            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(self.depth_output_dir2, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(self.normal_output_dir2, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(self.reflectance_output_dir2, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(self.illumination_output_dir2, '{}.mat'.format(name)),
                        {'result': illumination_pred})



if __name__ == "__main__":
    args = Options().parse()
    args.cuda = True
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.cuda)
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    torch.manual_seed(args.seed)
    args.lr = 1e-5
    print(args)

    trainer = Trainer(args)
    print(['Starting Epoch:', str(args.start_epoch)])
    print(['Total Epoches:', str(args.epochs)])
    for epoch in range(args.start_epoch, args.epochs):
        #trainer.test(epoch)
        trainer.training(epoch)
        if (epoch + 1) % 10 == 0:
            trainer.test(epoch)
