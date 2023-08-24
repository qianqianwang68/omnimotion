"""
This script filters the raft optical flow using cycle consistency and appearance consistency
checks (using dino features), and produces the following files:

raft masks: h x w x 3 for each pair of flows, first channel stores the mask for cycle consistency,
            second channel stores the mask for occlusion (i.e., regions that detected as occluded
            where the prediction is likely to be reliable using double cycle consistency checks).
count_maps: h x w for each frame (uint16), contains the number of valid correspondences for each pixel
            across all frames.
flow_stats.json: contains the total number of valid correspondences between each pair of frames.
"""

import json
import argparse
import os
import glob
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from chain_raft import gen_grid, normalize_coords
import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cuda'


def run_filtering(args):
    feature_name = 'dino'
    scene_dir = args.data_dir
    print('flitering raft optical flow for {}....'.format(scene_dir))

    img_files = sorted(glob.glob(os.path.join(scene_dir, 'color', '*')))
    num_imgs = len(img_files)
    pbar = tqdm(total=num_imgs * (num_imgs - 1))

    out_flow_stats_file = os.path.join(scene_dir, 'flow_stats.json')
    out_dir = os.path.join(scene_dir, 'raft_masks')
    os.makedirs(out_dir, exist_ok=True)

    count_out_dir = os.path.join(scene_dir, 'count_maps')
    os.makedirs(count_out_dir, exist_ok=True)

    h, w = imageio.imread(img_files[0]).shape[:2]
    grid = gen_grid(h, w, device=DEVICE).permute(2, 0, 1)[None]
    grid_normed = normalize_coords(grid.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]

    features = [torch.from_numpy(np.load(os.path.join(scene_dir, 'features', feature_name,
                                                      os.path.basename(img_file) + '.npy'))).float().to(DEVICE)
                for img_file in img_files]

    flow_stats = {}
    count_maps = np.zeros((num_imgs, h, w), dtype=np.uint16)
    for i in range(num_imgs):
        imgname_i = os.path.basename(img_files[i])
        feature_i = features[i].permute(2, 0, 1)[None]
        feature_i_sampled = F.grid_sample(feature_i, grid_normed[None], align_corners=True)[0].permute(1, 2, 0)

        for j in range(num_imgs):
            if i == j:
                continue
            frame_interval = abs(i - j)
            imgname_j = os.path.basename(img_files[j])
            flow_f = np.load(os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_i, imgname_j)))
            flow_f = torch.from_numpy(flow_f).float().permute(2, 0, 1)[None].cuda()
            flow_b = np.load(os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_j, imgname_i)))
            flow_b = torch.from_numpy(flow_b).float().permute(2, 0, 1)[None].cuda()

            coord2 = flow_f + grid
            coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]
            flow_21_sampled = F.grid_sample(flow_b, coord2_normed[None], align_corners=True)
            map_i = flow_f + flow_21_sampled
            fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
            mask_cycle = fb_discrepancy < args.cycle_th

            feature_j = features[j].permute(2, 0, 1)[None]
            feature_j_sampled = F.grid_sample(feature_j, coord2_normed[None], align_corners=True)[0].permute(1, 2, 0)
            feature_sim = torch.cosine_similarity(feature_i_sampled, feature_j_sampled, dim=-1)
            feature_mask = feature_sim > 0.5

            mask_cycle = mask_cycle * feature_mask if frame_interval >= 3 else mask_cycle

            # only keep correspondences for occluded pixels if the correspondences are
            # inconsistent in the first cycle but consistent in the second cycle
            # and if the two frames are adjacent enough (interval < 3)
            if frame_interval < 3:
                coord_21 = grid + map_i  # [1, 2, h, w]
                coord_21_normed = normalize_coords(coord_21.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]
                flow_22 = F.grid_sample(flow_f, coord_21_normed[None], align_corners=True)
                fbf_discrepancy = torch.norm((coord_21 + flow_22 - flow_f - grid).squeeze(), dim=0)
                mask_in_range = (coord2_normed.min(dim=-1)[0] >= -1) * (coord2_normed.max(dim=-1)[0] <= 1)
                mask_occluded = (fbf_discrepancy < args.cycle_th) * (fb_discrepancy > args.cycle_th * 1.5)
                mask_occluded *= mask_in_range
            else:
                mask_occluded = torch.zeros_like(mask_cycle)

            out_mask = torch.stack([mask_cycle, mask_occluded, torch.zeros_like(mask_cycle)], dim=-1).cpu().numpy()
            imageio.imwrite('{}/{}_{}.png'.format(out_dir, imgname_i, imgname_j), (255 * out_mask.astype(np.uint8)))

            if not imgname_i in flow_stats.keys():
                flow_stats[imgname_i] = {}
            flow_stats[imgname_i][imgname_j] = np.sum(out_mask).item()
            count_maps[i] += out_mask.sum(axis=-1).astype(np.uint16)
            pbar.update(1)

    pbar.close()
    with open(out_flow_stats_file, 'w') as fp:
        json.dump(flow_stats, fp)

    for i in range(num_imgs):
        img_name = os.path.basename(img_files[i])
        imageio.imwrite(os.path.join(count_out_dir, img_name.replace('.jpg', '.png')), count_maps[i])

    print('filtering raft optical flow for {} is done\n'.format(scene_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--cycle_th', type=float, default=3., help='threshold for cycle consistency error')
    args = parser.parse_args()

    run_filtering(args)
