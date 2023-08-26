import torch
import torch.nn as nn
import random
import ipdb
import os
from tqdm import tqdm
from functools import partial
import numpy as np
# np.warnings.filterwarnings('ignore')

import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import default_generator
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn

from PIL import Image

from datasets import datasets
import datasets.transforms as ext_transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

from resnet import FeatureCPResNet
from conformal import helper
from conformal.icp import IcpRegressor, RegressorNc, FeatRegressorNc
from conformal.icp import AbsErrorErrFunc, FeatErrorErrFunc
from conformal.utils import compute_coverage, WeightedMSE, seed_torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

def load_dataset():
    # Normalization from torchvision repo
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    cudnn.benchmark = True
    batch_size = 128
    num_calib = 5000

    # Get the conformal calibcration dataset
    imagenet_calib_data, imagenet_val_data = random_split(torchvision.datasets.ImageFolder('./datasets/data/imagenet_val', transform), [num_calib, 50000-num_calib])

    # Initialize loaders 
    calib_loader = DataLoader(imagenet_calib_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(imagenet_val_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return calib_loader, test_loader

def compute_coverage_set(s, y):
    n = y.shape[0]
    cvg = s[np.arange(n), y].mean()
    size = s.sum(1).mean()
    return cvg, size

def main(cal_loader, test_loader, args):
    resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1', progress=True).to(device)
    model = FeatureCPResNet(resnet)

    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)

    num_classes = len(cal_loader.dataset.dataset.classes)
    criterion = torch.nn.CrossEntropyLoss()

    # FeatureCP
    nc = FeatRegressorNc(model, num_classes=num_classes, criterion=criterion, inv_lr=args.feat_lr, inv_step=args.feat_step,
                         feat_norm=args.feat_norm, cert_optimizer=args.feat_opt, device=device)
    icp = IcpRegressor(nc)

    icp.calibrate_batch(cal_loader)

    # calculating the coverage of FCP
    in_coverage = icp.if_in_coverage_batch(test_loader, significance=alpha)
    coverage_fcp = np.sum(in_coverage) * 100 / len(in_coverage)

    test_sets = []
    all_y_test = []
    for x_test, y_test in tqdm(test_loader):
        s = icp.predict(x_test, significance=alpha)
        test_sets.append(s)
        all_y_test.append(y_test.cpu().numpy())

    test_sets = np.concatenate(test_sets, axis=0)
    all_y_test = np.concatenate(all_y_test, axis=0)
    # estimating the length of FCP

    cvg_out, length_fcp = compute_coverage_set(test_sets, all_y_test)
    
    print(f"coverage_fcp={coverage_fcp}, length_fcp={length_fcp}, Coverage_out={cvg_out}")
    
    assert False
    return coverage_fcp, length_fcp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=0, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument("--data", type=str, default="cityscapes", help="only support cityscapes now", choices=['cityscapes'])
    parser.add_argument("--data-seed", type=int, default=None, help="the random seed to split the calibration and test sets")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to the root directory of the selected dataset.")
    parser.add_argument("--workers", type=int, default=4, help="Number of subprocesses to use for data loading. Default: 4")

    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")

    parser.add_argument("--feat_opt", "--fo", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--feat_lr", "--fl", type=float, default=1e-3)
    parser.add_argument("--feat_step", "--fs", type=int, default=None)
    parser.add_argument("--feat_norm", "--fn", default=-1)
    parser.add_argument("--cert_method", "--cm", type=int, default=0, choices=[0, 1, 2, 3])

    parser.add_argument("--visualize", action="store_true", default=False, help="visualize the length in the image")
    args = parser.parse_args()

    fcp_coverage_list, fcp_length_list, cp_coverage_list, cp_length_list = [], [], [], []
    for seed in args.seed:
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
        device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

        if args.visualize:
            makedirs(f"./visualization/seed{seed}")

        nn_learn_func = torch.optim.Adam

        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.05
        # desired miscoverage error
        # alpha = 0.1
        alpha = args.alpha
        # used to determine the size of test set
        test_ratio = 0.2
        # seed for splitting the data in cross-validation.
        cv_random_state = 1

        if args.data.lower() == 'cityscapes':
            from datasets.cityscapes import Cityscapes as dataset
        else:
            # Should never happen...but just in case it does
            raise RuntimeError("\"{0}\" is not a supported dataset.".format(
                args.dataset))
        cal_loader, test_loader = load_dataset()
        print("Dataset: %s" % (args.data))

        coverage_fcp, length_fcp, coverage_cp, length_cp = \
            main(cal_loader, test_loader, args)
        fcp_coverage_list.append(coverage_fcp)
        fcp_length_list.append(length_fcp)
        cp_coverage_list.append(coverage_cp)
        cp_length_list.append(length_cp)

    print(f'FeatureCP coverage: {np.mean(fcp_coverage_list)} \pm {np.std(fcp_coverage_list)}',
          f'FeatureCP estimated length: {np.mean(fcp_length_list)} \pm {np.std(fcp_length_list)}')
    print(f'VanillaCP coverage: {np.mean(cp_coverage_list)} \pm {np.std(cp_coverage_list)}',
          f'VanillaCP length: {np.mean(cp_length_list)} \pm {np.std(cp_length_list)}')
