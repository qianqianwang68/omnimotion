"""
This script computes all pairwise RAFT optical flow fields
for each pair, we use previous flow as initialization to compute the current flow
"""

import sys

sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils.utils import InputPadder
import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def run_exhaustive_flow(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    data_dir = args.data_dir
    print('computing all pairwise optical flows for {}...'.format(data_dir))

    flow_out_dir = os.path.join(data_dir, 'raft_exhaustive')
    os.makedirs(flow_out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(data_dir, 'color', '*')))
    num_imgs = len(img_files)
    pbar = tqdm(total=num_imgs * (num_imgs - 1))
    with torch.no_grad():
        for i in range(num_imgs - 1):
            flow_low_prev = None
            for j in range(i + 1, num_imgs):
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low_prev)
                flow_up = padder.unpad(flow_up)

                flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                save_file = os.path.join(flow_out_dir,
                                         '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
                np.save(save_file, flow_up_np)
                flow_low_prev = flow_low
                pbar.update(1)

        for i in range(num_imgs - 1, 0, -1):
            flow_low_prev = None
            for j in range(i - 1, -1, -1):
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low_prev)
                flow_up = padder.unpad(flow_up)

                flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                save_file = os.path.join(flow_out_dir,
                                         '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
                np.save(save_file, flow_up_np)
                flow_low_prev = flow_low
                pbar.update(1)
        pbar.close()
        print('computing all pairwise optical flows for {} is done \n'.format(data_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    args = parser.parse_args()

    run_exhaustive_flow(args)


