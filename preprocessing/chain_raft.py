"""
chain cycle consistent correspondences to create longer and denser tracks.
the rules for chaining: only chain cycle consistent flows between adjacent frames,
and if the direct cycle consistent flow exists, the chained flows will be overwritten by the direct flows
which are considered to be more reliable, and the procedure then continues iteratively.
The chained flows will go through a final appearance check in both feature and RGB space
to reduce spurious correspondences. One can think of this process as augmenting the original
direct optical flows (which is unchanged) with some chained ones. This is optional and we found it
to help with the optimization process for sequences where the number of valid flows are very imbalanced
across regions.
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
import warnings

warnings.filterwarnings("ignore")


DEVICE = 'cuda'


def gen_grid(h, w, device, normalize=False, homogeneous=False):
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h, device=device)
        lin_x = torch.linspace(-1., 1., steps=w, device=device)
    else:
        lin_y = torch.arange(0, h, device=device)
        lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]


def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
    else:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.


def run(args):
    feature_name = 'dino'
    scene_dir = args.data_dir
    print('chaining raft optical flow for {}....'.format(scene_dir))

    img_files = sorted(glob.glob(os.path.join(scene_dir, 'color', '*')))
    num_imgs = len(img_files)
    pbar = tqdm(total=num_imgs*(num_imgs-1))

    out_dir = os.path.join(scene_dir, 'raft_exhaustive')
    out_mask_dir = os.path.join(scene_dir, 'raft_masks')
    count_out_dir = os.path.join(scene_dir, 'count_maps')
    out_flow_stats_file = os.path.join(scene_dir, 'flow_stats.json')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(count_out_dir, exist_ok=True)
    h, w = imageio.imread(img_files[0]).shape[:2]
    grid = gen_grid(h, w, 'cuda')[None]  # [b, h, w, 2]
    flow_stats = {}
    count_maps = np.zeros((num_imgs, h, w), dtype=np.uint16)

    images = [torch.from_numpy(imageio.imread(img_file) / 255.).float().permute(2, 0, 1)[None].to(DEVICE)
              for img_file in img_files]
    features = [torch.from_numpy(np.load(os.path.join(scene_dir, 'features', feature_name,
                                                      os.path.basename(img_file) + '.npy'))).float().to(DEVICE)
                for img_file in img_files]

    for i in range(num_imgs - 1):
        imgname_i = os.path.basename(img_files[i])
        imgname_i_plus_1 = os.path.basename(img_files[i + 1])
        start_flow_file = os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_i, imgname_i_plus_1))
        start_flow = np.load(start_flow_file)
        start_flow = torch.from_numpy(start_flow).float()[None].cuda()  # [b, h, w, 2]

        start_mask_file = start_flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        start_cycle_mask = imageio.imread(start_mask_file)[..., 0] > 0

        feature_i = features[i].permute(2, 0, 1)[None]
        feature_i = F.interpolate(feature_i, size=start_flow.shape[1:3], mode='bilinear')

        accumulated_flow = start_flow
        accumulated_cycle_mask = start_cycle_mask

        for j in range(i + 1, num_imgs):
            # # vis
            imgname_j = os.path.basename(img_files[j])
            direct_flow_file = os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_i, imgname_j))
            direct_flow = np.load(direct_flow_file)
            direct_flow = torch.from_numpy(direct_flow).float()[None].cuda()  # [b, h, w, 2]
            direct_mask_file = direct_flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
            direct_masks = imageio.imread(direct_mask_file)
            direct_cycle_mask = direct_masks[..., 0] > 0
            direct_occlusion_mask = direct_masks[..., 1] > 0
            direct_mask = direct_cycle_mask | direct_occlusion_mask
            direct_mask = torch.from_numpy(direct_mask)[None]

            accumulated_flow[direct_mask] = direct_flow[direct_mask]

            curr_coords = grid + accumulated_flow
            curr_coords_normed = normalize_coords(curr_coords, h, w)  # [b, h, w, 2]

            feature_j = features[j].permute(2, 0, 1)[None]
            feature_j_sampled = F.grid_sample(feature_j, curr_coords_normed, align_corners=True)
            feature_sim = torch.cosine_similarity(feature_i, feature_j_sampled, dim=1).squeeze(0).cpu().numpy()
            image_j_sampled = F.grid_sample(images[j], curr_coords_normed, align_corners=True).squeeze()
            rgb_sim = torch.norm(images[i] - image_j_sampled, dim=1).squeeze(0).cpu().numpy()

            accumulated_cycle_mask *= (feature_sim > 0.5) * (rgb_sim < 0.3)
            accumulated_cycle_mask[direct_cycle_mask] = True
            accumulated_cycle_mask[direct_occlusion_mask] = False

            np.save(os.path.join(out_dir, '{}_{}.npy'.format(imgname_i, imgname_j)), accumulated_flow[0].cpu().numpy())
            out_mask = np.concatenate([255 * accumulated_cycle_mask[..., None].astype(np.uint8),
                                       direct_masks[..., 1:]],
                                      axis=-1)
            imageio.imwrite('{}/{}_{}.png'.format(out_mask_dir, imgname_i, imgname_j), out_mask)
            count_maps[i] += (out_mask / 255).sum(axis=-1).astype(np.uint16)
            if not imgname_i in flow_stats.keys():
                flow_stats[imgname_i] = {}
            flow_stats[imgname_i][imgname_j] = int(np.sum(out_mask/255))

            pbar.update(1)

            if j == num_imgs - 1:
                continue

            imgname_j_plus_1 = os.path.basename(img_files[j + 1])
            flow_file = os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_j, imgname_j_plus_1))
            curr_flow = np.load(flow_file)
            curr_flow = torch.from_numpy(curr_flow).float()[None].cuda()  # [b, h, w, 2]
            curr_mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
            curr_cycle_mask = imageio.imread(curr_mask_file)[..., 0] > 0

            flow_curr_sampled = F.grid_sample(curr_flow.permute(0, 3, 1, 2), curr_coords_normed,
                                              align_corners=True).permute(0, 2, 3, 1)
            curr_cycle_mask_sampled = F.grid_sample(torch.from_numpy(curr_cycle_mask).float()[None, None].cuda(),
                                                    curr_coords_normed, align_corners=True).squeeze().cpu().numpy() == 1
            # update
            accumulated_flow += flow_curr_sampled
            accumulated_cycle_mask *= curr_cycle_mask_sampled

    for i in range(num_imgs - 1, 0, -1):
        imgname_i = os.path.basename(img_files[i])
        imgname_i_minus_1 = os.path.basename(img_files[i - 1])
        start_flow_file = os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_i, imgname_i_minus_1))
        start_flow = np.load(start_flow_file)
        start_flow = torch.from_numpy(start_flow).float()[None].cuda()  # [b, h, w, 2]

        start_mask_file = start_flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        start_cycle_mask = imageio.imread(start_mask_file)[..., 0] > 0

        feature_i = features[i].permute(2, 0, 1)[None]
        feature_i = F.interpolate(feature_i, size=start_flow.shape[1:3], mode='bilinear')

        accumulated_flow = start_flow
        accumulated_cycle_mask = start_cycle_mask

        for j in range(i - 1, -1, -1):
            # # vis
            imgname_j = os.path.basename(img_files[j])
            direct_flow_file = os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_i, imgname_j))
            direct_flow = np.load(direct_flow_file)
            direct_flow = torch.from_numpy(direct_flow).float()[None].cuda()  # [b, h, w, 2]
            direct_mask_file = direct_flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
            direct_masks = imageio.imread(direct_mask_file)
            direct_cycle_mask = direct_masks[..., 0] > 0
            direct_occlusion_mask = direct_masks[..., 1] > 0
            direct_mask = direct_cycle_mask | direct_occlusion_mask
            direct_mask = torch.from_numpy(direct_mask)[None]

            accumulated_flow[direct_mask] = direct_flow[direct_mask]

            curr_coords = grid + accumulated_flow
            curr_coords_normed = normalize_coords(curr_coords, h, w)  # [b, h, w, 2]

            feature_j = features[j].permute(2, 0, 1)[None]
            feature_j_sampled = F.grid_sample(feature_j, curr_coords_normed, align_corners=True)
            feature_sim = torch.cosine_similarity(feature_i, feature_j_sampled, dim=1).squeeze(0).cpu().numpy()
            image_j_sampled = F.grid_sample(images[j], curr_coords_normed, align_corners=True).squeeze()
            rgb_sim = torch.norm(images[i] - image_j_sampled, dim=1).squeeze(0).cpu().numpy()

            accumulated_cycle_mask *= (feature_sim > 0.5) * (rgb_sim < 0.3)
            accumulated_cycle_mask[direct_cycle_mask] = True
            accumulated_cycle_mask[direct_occlusion_mask] = False

            np.save(os.path.join(out_dir, '{}_{}.npy'.format(imgname_i, imgname_j)), accumulated_flow[0].cpu().numpy())
            out_mask = np.concatenate([255 * accumulated_cycle_mask[..., None].astype(np.uint8),
                                       direct_masks[..., 1:]],
                                      axis=-1)
            imageio.imwrite('{}/{}_{}.png'.format(out_mask_dir, imgname_i, imgname_j), out_mask)
            count_maps[i] += (out_mask / 255).sum(axis=-1).astype(np.uint16)
            if not imgname_i in flow_stats.keys():
                flow_stats[imgname_i] = {}
            flow_stats[imgname_i][imgname_j] = int(np.sum(out_mask / 255))

            pbar.update(1)

            if j == 0:
                continue

            imgname_j_minus_1 = os.path.basename(img_files[j - 1])
            flow_file = os.path.join(scene_dir, 'raft_exhaustive', '{}_{}.npy'.format(imgname_j, imgname_j_minus_1))
            curr_flow = np.load(flow_file)
            curr_flow = torch.from_numpy(curr_flow).float()[None].cuda()  # [b, h, w, 2]
            curr_mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
            curr_cycle_mask = imageio.imread(curr_mask_file)[..., 0] > 0

            flow_curr_sampled = F.grid_sample(curr_flow.permute(0, 3, 1, 2), curr_coords_normed,
                                              align_corners=True).permute(0, 2, 3, 1)
            curr_cycle_mask_sampled = F.grid_sample(torch.from_numpy(curr_cycle_mask).float()[None, None].cuda(),
                                                    curr_coords_normed, align_corners=True).squeeze().cpu().numpy() == 1
            # update
            accumulated_flow += flow_curr_sampled
            accumulated_cycle_mask *= curr_cycle_mask_sampled

    pbar.close()
    with open(out_flow_stats_file, 'w') as fp:
        json.dump(flow_stats, fp)

    for i in range(num_imgs):
        img_name = os.path.basename(img_files[i])
        imageio.imwrite(os.path.join(count_out_dir, img_name.replace('.jpg', '.png')), count_maps[i])

    print('chaining raft optical flow for {} is done'.format(scene_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')

    args = parser.parse_args()

    run(args)