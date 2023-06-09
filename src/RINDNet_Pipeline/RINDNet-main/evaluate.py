import argparse
import torch
import os
from tqdm import tqdm
import scipy.io as sio
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data import DataLoader
from modeling.rindnet import *

def main():
    parser = argparse.ArgumentParser(description="PyTorch Model Testing")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='bsds',
                        choices=['bsds'],
                        help='dataset name (default: pascal)')
    parser.add_argument("--data_path", type=str, help="path to the training data",
                        default="data/BSDS-RIND/BSDS-RIND/Augmentation/")
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--loss-type', type=str, default='attention',
                        choices=['ce', 'focal', 'attention'],
                        help='loss func type (default: ce)')
    # test hyper params
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--output-dir', type=str, default='run/rindnet/')
    parser.add_argument('--evaluate-model-path', type=str, default='run/rindnet/epoch_70_checkpoint.pth.tar')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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

    model_dict = torch.load(args.evaluate_model_path, map_location='cpu')
    checkpoint_dict = model_dict['state_dict']
    model = MyNet()
    model.load_state_dict(checkpoint_dict)
    model.cuda()
    model.eval()

    test_dataset = Mydataset(root_path=args.data_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    output_dir = args.output_dir
    depth_output_dir = os.path.join(output_dir, 'depth/mat')
    if not os.path.exists(depth_output_dir):
        os.makedirs(depth_output_dir)
    normal_output_dir = os.path.join(output_dir, 'normal/mat')
    if not os.path.exists(normal_output_dir):
        os.makedirs(normal_output_dir)
    reflectance_output_dir = os.path.join(output_dir, 'reflectance/mat')
    if not os.path.exists(reflectance_output_dir):
        os.makedirs(reflectance_output_dir)
    illumination_output_dir = os.path.join(output_dir, 'illumination/mat')
    if not os.path.exists(illumination_output_dir):
        os.makedirs(illumination_output_dir)

    with torch.no_grad():
        for batch_index, images in enumerate(tqdm(test_loader)):
            name = test_loader.dataset.images_name[batch_index]
            image = images.cuda()
            with torch.no_grad():
                unet1,out_depth,out_normal,out_reflectance,out_illumination = model(image)

            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(illumination_output_dir, '{}.mat'.format(name)),
                        {'result': illumination_pred})


if __name__ == "__main__":
    main()
