import glob
import os
import pdb
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from criterion import masked_mse_loss, masked_l1_loss, compute_depth_range_loss, lossfun_distortion
from networks.mfn import GaborNet
from networks.nvp_simplified import NVPSimplified
from kornia import morphology as morph


torch.manual_seed(1234)


def init_weights(m):
    # Initializes weights according to the DCGAN paper
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class BaseTrainer():
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device

        self.read_data()

        self.feature_mlp = GaborNet(in_size=1,
                                    hidden_size=256,
                                    n_layers=2,
                                    alpha=4.5,
                                    out_size=128).to(device)

        self.deform_mlp = NVPSimplified(n_layers=6,
                                        feature_dims=128,
                                        hidden_size=[256, 256, 256],
                                        proj_dims=256,
                                        code_proj_hidden_size=[],
                                        proj_type='fixed_positional_encoding',
                                        pe_freq=args.pe_freq,
                                        normalization=False,
                                        affine=args.use_affine,
                                        device=device).to(device)

        self.color_mlp = GaborNet(in_size=3,
                                  hidden_size=512,
                                  n_layers=3,
                                  alpha=3,
                                  out_size=4).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.feature_mlp.parameters(), 'lr': args.lr_feature},
            {'params': self.deform_mlp.parameters(), 'lr': args.lr_deform},
            {'params': self.color_mlp.parameters(), 'lr': args.lr_color},
        ])

        self.learnable_params = list(self.feature_mlp.parameters()) + \
                                list(self.deform_mlp.parameters()) + \
                                list(self.color_mlp.parameters())

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)
        seq_name = os.path.basename(args.data_dir.rstrip('/'))
        self.out_dir = os.path.join(args.save_dir, '{}_{}'.format(args.expname, seq_name))
        self.step = self.load_from_ckpt(self.out_dir,
                                        load_opt=self.args.load_opt,
                                        load_scheduler=self.args.load_scheduler)
        self.time_steps = torch.linspace(1, self.num_imgs, self.num_imgs, device=self.device)[:, None] / self.num_imgs

        if args.distributed:
            self.feature_mlp = torch.nn.parallel.DistributedDataParallel(
                self.feature_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
            self.deform_mlp = torch.nn.parallel.DistributedDataParallel(
                self.deform_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
            self.color_mlp = torch.nn.parallel.DistributedDataParallel(
                self.color_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

    def read_data(self):
        self.seq_dir = self.args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')

        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        self.num_imgs = min(self.args.num_imgs, len(img_files))
        self.img_files = img_files[:self.num_imgs]

        images = np.array([imageio.imread(img_file) / 255. for img_file in self.img_files])
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]

        mask_files = [img_file.replace('color', 'mask').replace('.jpg', '.png') for img_file in self.img_files]
        if os.path.exists(mask_files[0]):
            masks = np.array([imageio.imread(mask_file)[..., :3].sum(axis=-1) / 255.
                              if imageio.imread(mask_file).ndim == 3 else
                              imageio.imread(mask_file) / 255.
                              for mask_file in mask_files])
            self.masks = torch.from_numpy(masks).to(self.device) > 0.  # [n_imgs, h, w]
            self.with_mask = True
        else:
            self.masks = torch.ones(self.images.shape[:-1]).to(self.device) > 0.
            self.with_mask = False
        self.grid = util.gen_grid(self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()

    def project(self, x, return_depth=False):
        '''
        orthographic projection
        :param x: [..., 3]
        :param return_depth: if returning depth
        :return: pixel_coords in image space [..., 2], depth [..., 1]
        '''
        pixel_coords, depth = torch.split(x, dim=-1, split_size_or_sections=[2, 1])
        pixel_coords = util.denormalize_coords(pixel_coords, self.h, self.w)
        if return_depth:
            return pixel_coords, depth
        else:
            return pixel_coords

    def unproject(self, pixels, depths):
        '''
        orthographic unprojection
        :param pixels: [..., 2] pixel coordinates (unnormalized)
        :param depths: [..., 1]
        :return: 3d locations in normalized space [..., 3]
        '''
        assert pixels.shape[-1] in [2, 3]
        assert pixels.ndim == depths.ndim
        pixels = util.normalize_coords(pixels[..., :2], self.h, self.w)
        return torch.cat([pixels, depths], dim=-1)

    def get_in_range_mask(self, x, max_padding=0):
        mask = (x[..., 0] >= -max_padding) * \
               (x[..., 0] <= self.w - 1 + max_padding) * \
               (x[..., 1] >= -max_padding) * \
               (x[..., 1] <= self.h - 1 + max_padding)
        return mask

    def sample_3d_pts_for_pixels(self, pixels, return_depth=False, det=True, near_depth=None, far_depth=None):
        '''
        stratified sampling
        sample points on ray for each pixel
        :param pixels: [n_imgs, n_pts, 2]
        :param return_depth: True or False
        :param det: if deterministic
        :param near_depth: nearest depth
        :param far_depth: farthest depth
        :return: sampled 3d locations [n_imgs, n_pts, n_samples, 3]
        '''
        if near_depth is not None and far_depth is not None:
            assert pixels.shape[:-1] == near_depth.shape[:-1] == far_depth.shape[:-1]
            depths = near_depth + \
                     torch.linspace(0, 1., self.args.num_samples_ray, device=self.device)[None, None] \
                     * (far_depth - near_depth)
        else:
            depths = torch.linspace(self.args.min_depth, self.args.max_depth, self.args.num_samples_ray,
                                    device=self.device)
            pixels_shape = pixels.shape
            depths = depths[None, None, :].expand(*pixels_shape[:2], -1)
        if not det:
            # get intervals between samples
            mids = .5 * (depths[..., 1:] + depths[..., :-1])
            upper = torch.cat([mids, depths[..., -1:]], dim=-1)
            lower = torch.cat([depths[..., 0:1], mids], dim=-1)
            # uniform samples in those intervals
            t_rand = torch.rand_like(depths)
            depths = lower + (upper - lower) * t_rand  # [n_imgs, n_pts, n_samples]

        depths = depths[..., None]
        pixels_expand = pixels.unsqueeze(-2).expand(-1, -1, self.args.num_samples_ray, -1)

        x = self.unproject(pixels_expand, depths)  # [n_imgs, n_pts, n_samples, 3]
        if return_depth:
            return x, depths
        else:
            return x

    def get_prediction_one_way(self, x, id, inverse=False):
        '''
        mapping 3d points from local to canonical or from canonical to local (inverse=True)
        :param x: [n_imgs, n_pts, n_samples, 3]
        :param id: [n_imgs, ]
        :param inverse: True or False
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        t = self.time_steps[id]  # [n_imgs, 1]
        feature = self.feature_mlp(t)  # [n_imgs, n_feat]

        if inverse:
            if self.args.distributed:
                out = self.deform_mlp.module.inverse(t, feature, x)
            else:
                out = self.deform_mlp.inverse(t, feature, x)
        else:
            out = self.deform_mlp.forward(t, feature, x)

        return out  # [n_imgs, n_pts, n_samples, 3]

    def get_predictions(self, x1, id1, id2, return_canonical=False):
        '''
        mapping 3d points from one frame to another frame
        :param x1: [n_imgs, n_pts, n_samples, 3]
        :param id1: [n_imgs,]
        :param id2: [n_imgs,]
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        x1_canonical = self.get_prediction_one_way(x1, id1)
        x2_pred = self.get_prediction_one_way(x1_canonical, id2, inverse=True)
        if return_canonical:
            return x2_pred, x1_canonical
        else:
            return x2_pred  # [n_imgs, n_pts, n_samples, 3]

    def get_canonical_color_and_density(self, x_canonical, apply_contraction=True):
        def contraction(x):
            x_norm = x.norm(dim=-1)
            x_out = torch.zeros_like(x)
            x_out[x_norm <= 1] = x[x_norm <= 1]
            x_out[x_norm > 1] = ((2. - 1. / x_norm[..., None]) * (x / x_norm[..., None]))[x_norm > 1]
            return x_out

        if apply_contraction:
            x_canonical = contraction(x_canonical)
        out_canonical = self.color_mlp(x_canonical)
        color = torch.sigmoid(out_canonical[..., :3])  # [n_imgs, n_pts, n_samples, 3]
        density = F.softplus(out_canonical[..., -1] - 1.)
        return color, density

    def get_blending_weights(self, x_canonical):
        '''
        query the nerf network to color, density and blending weights
        :param x_canonical: input canonical 3D locations
        :return: dict containing colors, weights, alphas and rendered rgbs
        '''
        color, density = self.get_canonical_color_and_density(x_canonical)

        alpha = util.sigma2alpha(density)  # [n_imgs, n_pts, n_samples]

        # mask out the nearest 20% of samples. This trick may be helpful to avoid one local minimum solution
        # where surfaces that are not nearest to the camera are initialized at nearest depth planes
        if self.args.mask_near and self.step < 4000:
            mask = torch.ones_like(alpha)
            mask[..., :int(self.args.num_samples_ray * 0.2)] = 0
            alpha *= mask
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]  # [n_imgs, n_pts, n_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [n_imgs, n_pts, n_samples]

        weights = alpha * T  # [n_imgs, n_pts, n_samples]

        rendered_rgbs = torch.sum(weights.unsqueeze(-1) * color, dim=-2)  # [n_imgs, n_pts, 3]

        out = {'colors': color,
               'weights': weights,
               'alphas': alpha,
               'rendered_rgbs': rendered_rgbs,
               }
        return out

    def get_pred_rgbs_for_pixels(self, ids, pixels, return_weights=False):
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(pixels, return_depth=True)
        xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        out = self.get_blending_weights(xs_canonical_samples)
        blending_weights = out['weights']
        rendered_rgbs = out['rendered_rgbs']
        if return_weights:
            return rendered_rgbs, blending_weights  # [n_imgs, n_pts, 3], [n_imgs, n_pts, n_samples]
        else:
            return rendered_rgbs

    def get_pred_depths_for_pixels(self, ids, pixels):
        '''
        :param ids: list [n_imgs,]
        :param pixels: [n_imgs, n_pts, 2]
        :return: pred_depths: [n_imgs, n_pts, 1]
        '''
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(pixels, return_depth=True)
        xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        out = self.get_blending_weights(xs_canonical_samples)
        pred_depths = torch.sum(out['weights'].unsqueeze(-1) * pxs_depths_samples, dim=-2)
        return pred_depths  # [n_imgs, n_pts, 1]

    def get_pred_colors_and_depths_for_pixels(self, ids, pixels):
        '''
        :param ids: list [n_imgs,]
        :param pixels: [n_imgs, n_pts, 2]
        :return: pred_depths: [n_imgs, n_pts, 1]
        '''
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(pixels, return_depth=True)
        xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        out = self.get_blending_weights(xs_canonical_samples)
        pred_depths = torch.sum(out['weights'].unsqueeze(-1) * pxs_depths_samples, dim=-2)
        rendered_rgbs = out['rendered_rgbs']
        return rendered_rgbs, pred_depths  # [n_imgs, n_pts, 1]

    def compute_depth_consistency_loss(self, proj_depths, pred_depths, visibilities, normalize=True):
        '''
        :param proj_depths: [n_imgs, n_pts, 1]
        :param pred_depths: [n_imgs, n_pts, 1]
        :param visibilities: [n_imgs, n_pts, 1]
        :return: depth loss
        '''
        if normalize:
            mse_error = torch.mean((proj_depths - pred_depths) ** 2 * visibilities) / (torch.mean(visibilities) + 1e-6)
        else:
            mse_error = torch.mean((proj_depths - pred_depths) ** 2 * visibilities)
        return mse_error

    def get_correspondences_for_pixels(self, ids1, px1s, ids2,
                                       return_depth=False,
                                       use_max_loc=False):
        '''
        get correspondences for pixels in one frame to another frame
        :param ids1: [num_imgs]
        :param px1s: [num_imgs, num_pts, 2]
        :param ids2: [num_imgs]
        :param return_depth: if returning the depth of the mapped point in the target frame
        :param use_max_loc: if using only the sample with the maximum blending weight to
                            compute the corresponding location rather than doing over composition.
                            set to True leads to better results on occlusion boundaries,
                            by default it is set to True for inference.

        :return: px2s_pred: [num_imgs, num_pts, 2], and optionally depth: [num_imgs, num_pts, 1]
        '''
        # [n_pair, n_pts, n_samples, 3]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s)
        x2s_proj_samples, xs_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)
        out = self.get_blending_weights(xs_canonical_samples)  # [n_imgs, n_pts, n_samples]
        if use_max_loc:
            blending_weights = out['weights']
            indices = torch.max(blending_weights, dim=-1, keepdim=True)[1]
            x2s_pred = torch.gather(x2s_proj_samples, 2, indices[..., None].repeat(1, 1, 1, 3)).squeeze(-2)
            return self.project(x2s_pred, return_depth=return_depth)
        else:
            x2s_pred = torch.sum(out['weights'].unsqueeze(-1) * x2s_proj_samples, dim=-2)
            return self.project(x2s_pred, return_depth=return_depth)

    def get_correspondences_and_occlusion_masks_for_pixels(self, ids1, px1s, ids2,
                                                           return_depth=False,
                                                           use_max_loc=False):
        px2s_pred, depth_proj = self.get_correspondences_for_pixels(ids1, px1s, ids2,
                                                                    return_depth=True,
                                                                    use_max_loc=use_max_loc)
        px2s_pred_samples, px2s_pred_depths_samples = self.sample_3d_pts_for_pixels(px2s_pred, return_depth=True)
        xs_canonical_samples = self.get_prediction_one_way(px2s_pred_samples, ids2)
        out = self.get_blending_weights(xs_canonical_samples)
        weights = out['weights']
        eps = 1.1 * (self.args.max_depth - self.args.min_depth) / self.args.num_samples_ray
        mask_zero = px2s_pred_depths_samples.squeeze(-1) >= (depth_proj.expand(-1, -1, self.args.num_samples_ray) - eps)
        weights[mask_zero] = 0.
        occlusion_score = weights.sum(dim=-1, keepdim=True)
        if return_depth:
            return px2s_pred, occlusion_score, depth_proj
        else:
            return px2s_pred, occlusion_score

    def compute_scene_flow_smoothness_loss(self, ids, xs):
        mask_valid = (ids >= 1) * (ids < self.num_imgs - 1)
        ids = ids[mask_valid]
        if len(ids) == 0:
            return torch.tensor(0.)
        xs = xs[mask_valid]
        ids_prev = ids - 1
        ids_after = ids + 1
        xs_prev_after = self.get_predictions(torch.cat([xs, xs], dim=0),
                                             np.concatenate([ids, ids]),
                                             np.concatenate([ids_prev, ids_after]))
        xs_prev, xs_after = torch.split(xs_prev_after, split_size_or_sections=len(xs), dim=0)
        scene_flow_prev = xs - xs_prev
        scene_flow_after = xs_after - xs
        loss = masked_l1_loss(scene_flow_prev, scene_flow_after)
        return loss

    def canonical_sphere_loss(self, xs_canonical_samples, radius=1.):
        ''' encourage mapped locations to be within a (unit) sphere '''
        xs_canonical_norm = (xs_canonical_samples ** 2).sum(dim=-1)
        if (xs_canonical_norm >= 1.).any():
            canonical_unit_sphere_loss = ((xs_canonical_norm[xs_canonical_norm >= radius] - 1) ** 2).mean()
        else:
            canonical_unit_sphere_loss = torch.tensor(0.)
        return canonical_unit_sphere_loss

    def gradient_loss(self, pred, gt, weight=None):
        '''
        coordinate
        :param pred: [n_imgs, n_pts, 2] or [n_pts, 2]
        :param gt:
        :return:
        '''
        pred_grad = pred[..., 1:, :] - pred[..., :-1, :]
        gt_grad = gt[..., 1:, :] - gt[..., :-1, :]
        if weight is not None:
            weight_grad = weight[..., 1:, :] * weight[..., :-1, :]
        else:
            weight_grad = None
        loss = masked_l1_loss(pred_grad, gt_grad, weight_grad)
        return loss

    def compute_all_losses(self,
                           batch,
                           w_rgb=1,
                           w_depth_range=10,
                           w_distortion=1.,
                           w_scene_flow_smooth=10.,
                           w_canonical_unit_sphere=0.,
                           w_flow_grad=0.01,
                           write_logs=True,
                           return_data=False,
                           log_prefix='loss',
                           ):

        depth_min_th = self.args.min_depth
        depth_max_th = self.args.max_depth
        max_padding = self.args.max_padding

        ids1 = batch['ids1'].numpy()
        ids2 = batch['ids2'].numpy()
        px1s = batch['pts1'].to(self.device)
        px2s = batch['pts2'].to(self.device)
        gt_rgb1 = batch['gt_rgb1'].to(self.device)
        weights = batch['weights'].to(self.device)
        num_pts = px1s.shape[1]

        # [n_pair, n_pts, n_samples, 3]
        x1s_samples, px1s_depths_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=True, det=False)
        x2s_proj_samples, x1s_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)
        out = self.get_blending_weights(x1s_canonical_samples)
        blending_weights1 = out['weights']
        alphas1 = out['alphas']
        pred_rgb1 = out['rendered_rgbs']

        mask = (x2s_proj_samples[..., -1] >= depth_min_th) * (x2s_proj_samples[..., -1] <= depth_max_th)
        blending_weights1 = blending_weights1 * mask.float()
        x2s_pred = torch.sum(blending_weights1.unsqueeze(-1) * x2s_proj_samples, dim=-2)

        # [n_imgs, n_pts, n_samples, 2]
        px2s_proj_samples, px2s_proj_depth_samples = self.project(x2s_proj_samples, return_depth=True)
        px2s_proj, px2s_proj_depths = self.project(x2s_pred, return_depth=True)

        mask = self.get_in_range_mask(px2s_proj, max_padding)
        rgb_mask = self.get_in_range_mask(px1s)

        if mask.sum() > 0:
            loss_rgb = F.mse_loss(pred_rgb1[rgb_mask], gt_rgb1[rgb_mask])
            loss_rgb_grad = self.gradient_loss(pred_rgb1[rgb_mask], gt_rgb1[rgb_mask])

            optical_flow_loss = masked_l1_loss(px2s_proj[mask], px2s[mask], weights[mask], normalize=False)
            optical_flow_grad_loss = self.gradient_loss(px2s_proj[mask], px2s[mask], weights[mask])
        else:
            loss_rgb = loss_rgb_grad = optical_flow_loss = optical_flow_grad_loss = torch.tensor(0.)

        # mapped depth should be within the predefined range
        depth_range_loss = compute_depth_range_loss(px2s_proj_depth_samples, depth_min_th, depth_max_th)

        # distortion loss to remove floaters
        t = torch.cat([px1s_depths_samples[..., 0], px1s_depths_samples[..., 0][..., -1:]], dim=-1)
        distortion_loss = lossfun_distortion(t, blending_weights1)

        # scene flow smoothness
        # only apply to 25% of points to reduce cost
        scene_flow_smoothness_loss = self.compute_scene_flow_smoothness_loss(ids1, x1s_samples[:, :int(num_pts / 4)])

        # loss for mapped points to stay within canonical sphere
        canonical_unit_sphere_loss = self.canonical_sphere_loss(x1s_canonical_samples)

        loss = optical_flow_loss + \
               w_rgb * (loss_rgb + loss_rgb_grad) + \
               w_depth_range * depth_range_loss + \
               w_distortion * distortion_loss + \
               w_scene_flow_smooth * scene_flow_smoothness_loss + \
               w_canonical_unit_sphere * canonical_unit_sphere_loss + \
               w_flow_grad * optical_flow_grad_loss

        if write_logs:
            self.scalars_to_log['{}/Loss'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/loss_flow'.format(log_prefix)] = optical_flow_loss.item()
            self.scalars_to_log['{}/loss_rgb'.format(log_prefix)] = loss_rgb.item()
            self.scalars_to_log['{}/loss_depth_range'.format(log_prefix)] = depth_range_loss.item()
            self.scalars_to_log['{}/loss_distortion'.format(log_prefix)] = distortion_loss.item()
            self.scalars_to_log['{}/loss_scene_flow_smoothness'.format(log_prefix)] = scene_flow_smoothness_loss.item()
            self.scalars_to_log['{}/loss_canonical_unit_sphere'.format(log_prefix)] = canonical_unit_sphere_loss.item()
            self.scalars_to_log['{}/loss_flow_gradient'.format(log_prefix)] = optical_flow_grad_loss.item()

        data = {'ids1': ids1,
                'ids2': ids2,
                'x1s': x1s_samples,
                'x2s_pred': x2s_pred,
                'xs_canonical': x1s_canonical_samples,
                'mask': mask,
                'px2s_proj': px2s_proj,
                'px2s_proj_depths': px2s_proj_depths,
                'blending_weights': blending_weights1,
                'alphas': alphas1,
                't': t
                }
        if return_data:
            return loss, data
        else:
            return loss

    def weight_scheduler(self, step, start_step, w, min_weight, max_weight):
        if step <= start_step:
            weight = 0.0
        else:
            weight = w * (step - start_step)
        weight = np.clip(weight, a_min=min_weight, a_max=max_weight)
        return weight

    def train_one_step(self, step, batch):
        self.step = step
        start = time.time()
        self.scalars_to_log = {}

        self.optimizer.zero_grad()
        w_rgb = self.weight_scheduler(step, 0, 1./5000, 0, 10)
        w_flow_grad = self.weight_scheduler(step, 0, 1./500000, 0, 0.1)
        w_distortion = self.weight_scheduler(step, 40000, 1./2000, 0, 10)
        w_scene_flow_smooth = 20.

        loss, flow_data = self.compute_all_losses(batch,
                                                  w_rgb=w_rgb,
                                                  w_scene_flow_smooth=w_scene_flow_smooth,
                                                  w_distortion=w_distortion,
                                                  w_flow_grad=w_flow_grad,
                                                  return_data=True)

        if torch.isnan(loss):
            pdb.set_trace()

        loss.backward()

        is_break = False
        for p in self.deform_mlp.parameters():
            if torch.isnan(p.data).any() or torch.isnan(p.grad).any():
                is_break = True

        for p in self.feature_mlp.parameters():
            if torch.isnan(p.data).any() or torch.isnan(p.grad).any():
                is_break = True

        for p in self.color_mlp.parameters():
            if torch.isnan(p.data).any() or torch.isnan(p.grad).any():
                is_break = True

        if is_break:
            pdb.set_trace()

        if self.args.grad_clip > 0:
            for param in self.learnable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(param, self.args.grad_clip)
                if grad_norm > self.args.grad_clip:
                    print("Warning! Clip gradient from {} to {}".format(grad_norm, self.args.grad_clip))

        self.optimizer.step()
        self.scheduler.step()

        self.scalars_to_log['lr'] = self.optimizer.param_groups[0]['lr']

        self.scalars_to_log['time'] = time.time() - start
        self.ids1 = flow_data['ids1']
        self.ids2 = flow_data['ids2']

    def sample_pts_within_mask(self, mask, num_pts, return_normed=False, seed=None,
                               use_mask=False, reverse_mask=False, regular=False, interval=10):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if use_mask:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(self.grid[..., 0], dtype=torch.bool)

        if regular:
            coords = self.grid[::interval, ::interval, :2][mask[::interval, ::interval]]
        else:
            coords_valid = self.grid[mask][..., :2]
            rand_inds = rng.choice(len(coords_valid), num_pts, replace=(num_pts > len(coords_valid)))
            coords = coords_valid[rand_inds]

        coords_normed = util.normalize_coords(coords, self.h, self.w)
        if return_normed:
            return coords, coords_normed
        else:
            return coords  # [num_pts, 2]

    def generate_uniform_3d_samples(self, num_pts, radius=2):
        num_pts = int(num_pts)
        pts = 2. * torch.rand(num_pts * 2, 3, device=self.device) - 1.  # [-1, 1]^3
        pts_norm = torch.norm(pts, dim=-1)
        pts = pts[pts_norm < 1.]
        rand_ids = np.random.choice(len(pts), num_pts, replace=len(pts) < num_pts)
        pts = pts[rand_ids]
        pts *= radius
        return pts

    def get_canonical_uvw_from_frames(self, num_pts_per_frame=10000):
        uvws = []
        for i in range(self.num_imgs):
            pixels_normed = 2 * torch.rand(num_pts_per_frame, 2, device=self.device) - 1.
            pixels = util.denormalize_coords(pixels_normed, self.h, self.w)[None]
            pixel_samples = self.sample_3d_pts_for_pixels(pixels, det=False)
            with torch.no_grad():
                uvw = self.get_prediction_one_way(pixel_samples, [i])[0]
            uvws.append(uvw.reshape(-1, 3))
        uvws = torch.cat(uvws, dim=0)
        return uvws

    def save_canonical_rgba_volume(self, num_pts, sample_points_from_frames=False):
        save_dir = os.path.join(self.out_dir, 'pcd')
        os.makedirs(save_dir, exist_ok=True)
        chunk_size = self.args.chunk_size
        if sample_points_from_frames:
            num_pts_per_frame = num_pts // (self.args.num_imgs * self.args.num_samples_ray)
            uvw = self.get_canonical_uvw_from_frames(num_pts_per_frame)
            suffix = '_frames'
            apply_contraction = True
        else:
            uvw = self.generate_uniform_3d_samples(num_pts, radius=1)
            suffix = ''
            apply_contraction = False
        uvw_np = uvw.cpu().numpy()
        rgbas = []
        for chunk in torch.split(uvw, chunk_size, dim=0):
            with torch.no_grad():
                color, density = self.get_canonical_color_and_density(chunk, apply_contraction=apply_contraction)
            alpha = util.sigma2alpha(density)
            rgba = torch.cat([color, alpha[..., None]], dim=-1)
            rgbas.append(rgba.cpu().numpy())
        rgbas = np.concatenate(rgbas, axis=0)
        out = np.ascontiguousarray(np.concatenate([uvw_np, rgbas], axis=-1))
        np.save(os.path.join(save_dir, '{:06d}{}.npy'.format(self.step, suffix)), out)

    def vis_pairwise_correspondences(self, ids=None, num_pts=200, use_mask=False, use_max_loc=True,
                                     reverse_mask=False, regular=True, interval=20):
        if ids is not None:
            id1, id2 = ids
        else:
            id1 = self.ids1[0]
            id2 = self.ids2[0]

        px1s = self.sample_pts_within_mask(self.masks[id1], num_pts, seed=1234,
                                           use_mask=use_mask, reverse_mask=reverse_mask,
                                           regular=regular, interval=interval)
        num_pts = len(px1s)

        with torch.no_grad():
            px2s_pred, occlusion_score = \
                self.get_correspondences_and_occlusion_masks_for_pixels([id1], px1s[None], [id2],
                                                                        use_max_loc=use_max_loc)
            px2s_pred = px2s_pred[0]
            mask = occlusion_score > self.args.occlusion_th

        kp1 = px1s.detach().cpu().numpy()
        kp2 = px2s_pred.detach().cpu().numpy()
        img1 = self.images[id1].cpu().numpy()
        img2 = self.images[id2].cpu().numpy()
        mask = mask[0].squeeze(-1).cpu().numpy()
        out = util.drawMatches(img1, img2, kp1, kp2, num_vis=num_pts, mask=mask)
        out = cv2.putText(out, str(id2 - id1), org=(30, 50), fontScale=1, color=(255, 255, 255),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)
        out = util.uint82float(out)
        return out

    def plot_correspondences_for_pixels(self, query_kpt, query_id, num_pts=200,
                                        vis_occlusion=False,
                                        occlusion_th=0.95,
                                        use_max_loc=False,
                                        radius=2,
                                        return_kpts=False):
        frames = []
        kpts = []
        with torch.no_grad():
            img_query = self.images[query_id].cpu().numpy()
            for id in range(0, self.num_imgs):
                if vis_occlusion:
                    if id == query_id:
                        kp_i = query_kpt
                        occlusion_score = torch.zeros_like(query_kpt[..., :1])
                    else:
                        kp_i, occlusion_score = \
                            self.get_correspondences_and_occlusion_masks_for_pixels([query_id], query_kpt[None], [id],
                                                                                    use_max_loc=use_max_loc)
                        kp_i = kp_i[0]
                        occlusion_score = occlusion_score[0]

                    mask = occlusion_score > occlusion_th
                    kp_i = torch.cat([kp_i, mask.float()], dim=-1)
                    mask = mask.squeeze(-1).cpu().numpy()
                else:
                    if id == query_id:
                        kp_i = query_kpt
                    else:
                        kp_i = self.get_correspondences_for_pixels([query_id], query_kpt[None], [id],
                                                                   use_max_loc=use_max_loc)[0]
                    mask = None
                img_i = self.images[id].cpu().numpy()
                out = util.drawMatches(img_query, img_i, query_kpt.cpu().numpy(), kp_i.cpu().numpy(),
                                       num_vis=num_pts, mask=mask, radius=radius)
                frames.append(out)
                kpts.append(kp_i)
        kpts = torch.stack(kpts, dim=0)
        if return_kpts:
            return frames, kpts
        return frames

    def eval_video_correspondences(self, query_id, pts=None, num_pts=200, seed=1234, use_mask=False,
                                   mask=None, reverse_mask=False, vis_occlusion=False, occlusion_th=0.99,
                                   use_max_loc=False, regular=True,
                                   interval=10, radius=2, return_kpts=False):
        with torch.no_grad():
            if mask is not None:
                mask = torch.from_numpy(mask).bool().to(self.device)
            else:
                mask = self.masks[query_id]

            if pts is None:
                x_0 = self.sample_pts_within_mask(mask, num_pts, seed=seed, use_mask=use_mask,
                                                  reverse_mask=reverse_mask, regular=regular, interval=interval)
                num_pts = 1e7 if regular else num_pts
            else:
                x_0 = torch.from_numpy(pts).float().to(self.device)
            return self.plot_correspondences_for_pixels(x_0, query_id, num_pts=num_pts,
                                                        vis_occlusion=vis_occlusion,
                                                        occlusion_th=occlusion_th,
                                                        use_max_loc=use_max_loc,
                                                        radius=radius, return_kpts=return_kpts)

    def get_pred_depth_maps(self, ids, chunk_size=40000):
        grid = self.grid[..., :2].reshape(-1, 2)
        pred_depths = []
        for id in ids:
            depth_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                depths_chunk = self.get_pred_depths_for_pixels([id], coords[None])[0]
                depths_chunk = torch.nan_to_num(depths_chunk)
                depth_map.append(depths_chunk)
            depth_map = torch.cat(depth_map, dim=0).reshape(self.h, self.w)
            pred_depths.append(depth_map)
        pred_depths = torch.stack(pred_depths, dim=0)
        return pred_depths  # [n, h, w]

    def get_pred_imgs(self, ids, chunk_size=40000, return_weights_stats=False):
        grid = self.grid[..., :2].reshape(-1, 2)
        pred_rgbs = []
        weights_stats = []
        for id in ids:
            rgb = []
            weights_stat = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                if return_weights_stats:
                    rgbs_chunk, weights_stats_chunk = self.get_pred_rgbs_for_pixels([id], coords[None],
                                                                                    return_weights=return_weights_stats)
                    weights_sum = weights_stats_chunk[0].sum(dim=-1)
                    weights_max = weights_stats_chunk[0].max(dim=-1)[0]
                    weights_stats_chunk = torch.stack([weights_sum, weights_max], dim=-1)
                    weights_stat.append(weights_stats_chunk)
                else:
                    rgbs_chunk = self.get_pred_rgbs_for_pixels([id], coords[None])
                rgb.append(rgbs_chunk[0])
            img = torch.cat(rgb, dim=0).reshape(self.h, self.w, 3)
            pred_rgbs.append(img)
            if return_weights_stats:
                weights_stats.append(torch.cat(weights_stat, dim=0).reshape(self.h, self.w, 2))

        pred_rgbs = torch.stack(pred_rgbs, dim=0)
        if return_weights_stats:
            weights_stats = torch.stack(weights_stats, dim=0)  # [n, h, w, 2]
            return pred_rgbs, weights_stats
        return pred_rgbs  # [n, h, w, 3]

    def get_pred_color_and_depth_maps(self, ids, chunk_size=40000):
        grid = self.grid[..., :2].reshape(-1, 2)
        pred_rgbs = []
        pred_depths = []
        for id in ids:
            rgb = []
            depth_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                rgbs_chunk = self.get_pred_rgbs_for_pixels([id], coords[None])
                rgb.append(rgbs_chunk[0])
                depths_chunk = self.get_pred_depths_for_pixels([id], coords[None])
                depths_chunk = torch.nan_to_num(depths_chunk)
                depth_map.append(depths_chunk[0])

            img = torch.cat(rgb, dim=0).reshape(self.h, self.w, 3)
            pred_rgbs.append(img)
            depth_map = torch.cat(depth_map, dim=0).reshape(self.h, self.w)
            pred_depths.append(depth_map)

        pred_rgbs = torch.stack(pred_rgbs, dim=0)
        pred_depths = torch.stack(pred_depths, dim=0)
        return pred_rgbs, pred_depths  # [n, h, w, 3/1]

    def get_pred_flows(self, ids1, ids2, chunk_size=40000, use_max_loc=False, return_original=False):
        grid = self.grid[..., :2].reshape(-1, 2)
        flows = []
        for id1, id2 in zip(ids1, ids2):
            flow_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                with torch.no_grad():
                    flows_chunk = self.get_correspondences_for_pixels([id1], coords[None], [id2],
                                                                      use_max_loc=use_max_loc)[0]
                    flow_map.append(flows_chunk)
            flow_map = torch.cat(flow_map, dim=0).reshape(self.h, self.w, 2)
            flow_map = (flow_map - self.grid[..., :2]).cpu().numpy()
            flows.append(flow_map)
        flows = np.stack(flows, axis=0)
        flow_imgs = util.flow_to_image(flows)
        if return_original:
            return flow_imgs, flows
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra

    def get_pred_flows_and_occlusions(self, ids1, ids2, chunk_size=40000, return_original=False):
        grid = self.grid[..., :2].reshape(-1, 2)
        flows = []
        for id1, id2 in zip(ids1, ids2):
            flow_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                with torch.no_grad():
                    flows_chunk, occlusion_chunk = self.get_correspondences_and_occlusion_masks_for_pixels([id1],
                                                                                                           coords[None],
                                                                                                           [id2])
                    flows_chunk = torch.cat([flows_chunk[0], occlusion_chunk[0].float()], dim=-1)
                    flow_map.append(flows_chunk)
            flow_map = torch.cat(flow_map, dim=0).reshape(self.h, self.w, 3)
            flow_map[..., :2] -= self.grid[..., :2]
            flow_map = flow_map.cpu().numpy()
            flows.append(flow_map)
        flows = np.stack(flows, axis=0)
        flow_imgs = util.flow_to_image(flows[..., :2])
        if return_original:
            return flow_imgs, flows
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra

    def render_color_and_depth_videos(self, start_id, end_id, chunk_size=40000, colorize=True):
        depths_np = []
        colors_np = []
        for id in range(start_id, end_id):
            with torch.no_grad():
                color, depth = self.get_pred_color_and_depth_maps([id], chunk_size=chunk_size)
                colors_np.append(color.cpu().numpy())
                depths_np.append(depth.cpu().numpy())

        colors_np = np.concatenate(colors_np, axis=0)
        depths_np = np.concatenate(depths_np, axis=0)
        depths_vis_min, depths_vis_max = depths_np.min(), depths_np.max()
        depths_vis = (depths_np - depths_vis_min) / (depths_vis_max - depths_vis_min)
        if colorize:
            depths_vis_color = []
            for depth_vis in depths_vis:
                depth_vis_color = util.colorize_np(depth_vis, range=(0, 1))
                depths_vis_color.append(depth_vis_color)
            depths_vis_color = np.stack(depths_vis_color, axis=0)
        else:
            depths_vis_color = depths_vis
        colors_np = (255 * colors_np).astype(np.uint8)
        depths_vis_color = (255 * depths_vis_color).astype(np.uint8)
        return colors_np, depths_vis_color

    def log(self, writer, step):
        if self.args.local_rank == 0:
            if step % self.args.i_print == 0:
                logstr = '{}_{} | step: {} |'.format(self.args.expname, self.seq_name, step)
                for k in self.scalars_to_log.keys():
                    logstr += ' {}: {:.6f}'.format(k, self.scalars_to_log[k])
                    if k != 'time':
                        writer.add_scalar(k, self.scalars_to_log[k], step)
                print(logstr)

            if step % self.args.i_img == 0:
                # flow
                flows = self.get_pred_flows(self.ids1[0:1], self.ids2[0:1], chunk_size=self.args.chunk_size)[0]
                writer.add_image('flow', flows, step, dataformats='HWC')

                # correspondences
                out_trained = self.vis_pairwise_correspondences()
                out_fix_10 = self.vis_pairwise_correspondences(ids=(0, min(self.num_imgs // 10, 10)))
                out_fix_half = self.vis_pairwise_correspondences(ids=(0, self.num_imgs // 2))
                out_fix_full = self.vis_pairwise_correspondences(ids=(0, self.num_imgs - 1))

                writer.add_image('correspondence/trained', out_trained, step, dataformats='HWC')
                writer.add_image('correspondence/fix_10', out_fix_10, step, dataformats='HWC')
                writer.add_image('correspondence/fix_half', out_fix_half, step, dataformats='HWC')
                writer.add_image('correspondence/fix_whole', out_fix_full, step, dataformats='HWC')

                # write predicted depths
                ids = np.concatenate([self.ids1[0:1], self.ids2[0:1]])
                with torch.no_grad():
                    pred_depths = self.get_pred_depth_maps(ids, chunk_size=self.args.chunk_size).cpu()  # [n, h, w]
                    pred_imgs, weights_stats = self.get_pred_imgs(ids, return_weights_stats=True,
                                                                  chunk_size=self.args.chunk_size)  # [n, h, w, 3/2]
                    pred_imgs = pred_imgs.cpu()
                    weights_stats = weights_stats.cpu()

                # write depth maps
                pred_depths_cat = pred_depths.permute(1, 0, 2).reshape(self.h, -1)
                min_depth = pred_depths_cat.min().item()
                max_depth = pred_depths_cat.max().item()

                pred_depths_vis = util.colorize(pred_depths_cat, range=(min_depth, max_depth), append_cbar=True)
                pred_depths_vis = F.interpolate(pred_depths_vis.permute(2, 0, 1)[None], scale_factor=0.5, mode='area')
                writer.add_image('depth', pred_depths_vis, step, dataformats='NCHW')

                # write gt and predicted rgbs
                gt_imgs = self.images[ids].cpu()
                imgs_vis = torch.cat([gt_imgs, pred_imgs], dim=1)
                imgs_vis = F.interpolate(imgs_vis.permute(0, 3, 1, 2), scale_factor=0.5, mode='area')
                writer.add_images('images', imgs_vis, step, dataformats='NCHW')

                # write weight statistics, first row: sum, second row: max
                weights_stats = weights_stats.permute(3, 1, 0, 2).reshape(len(ids) * self.h, -1)
                weights_stats_vis = util.colorize(weights_stats, range=(0, 1), append_cbar=True)
                weights_stats_vis = F.interpolate(weights_stats_vis.permute(2, 0, 1)[None], scale_factor=0.5,
                                                  mode='area')
                writer.add_image('weight_stats', weights_stats_vis, step, dataformats='NCHW')

            if step % self.args.i_weight == 0 and step > 0:
                # save checkpoints
                os.makedirs(self.out_dir, exist_ok=True)
                print('Saving checkpoints at {} to {}...'.format(step, self.out_dir))
                fpath = os.path.join(self.out_dir, 'model_{:06d}.pth'.format(step))
                self.save_model(fpath)

                vis_dir = os.path.join(self.out_dir, 'vis')
                os.makedirs(vis_dir, exist_ok=True)
                print('saving visualizations to {}...'.format(vis_dir))
                if self.with_mask:
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            use_mask=True,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_foreground_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            use_mask=True,
                                                                            reverse_mask=True,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_background_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                else:
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                color_frames, depth_frames = self.render_color_and_depth_videos(0, self.num_imgs,
                                                                                chunk_size=self.args.chunk_size)
                imageio.mimwrite(os.path.join(vis_dir, '{}_depth_{:06d}.mp4'.format(self.seq_name, step)), depth_frames,
                                 quality=8, fps=10)
                imageio.mimwrite(os.path.join(vis_dir, '{}_color_{:06d}.mp4'.format(self.seq_name, step)), color_frames,
                                 quality=8, fps=10)

                ids1 = np.arange(self.num_imgs)
                ids2 = ids1 + 1
                ids2[-1] -= 2
                pred_optical_flows_vis, pred_optical_flows = self.get_pred_flows(ids1, ids2,
                                                                                 use_max_loc=self.args.use_max_loc,
                                                                                 chunk_size=self.args.chunk_size,
                                                                                 return_original=True
                                                                                 )
                imageio.mimwrite(os.path.join(vis_dir, '{}_flow_{:06d}.mp4'.format(self.seq_name, step)),
                                 pred_optical_flows_vis[:-1],
                                 quality=8, fps=10)

            if self.args.use_error_map and (step % self.args.i_cache == 0) and (step > 0):
                flow_save_dir = os.path.join(self.out_dir, 'flow')
                os.makedirs(flow_save_dir, exist_ok=True)
                flow_errors = []
                for i, (id1, id2) in enumerate(zip(ids1, ids2)):
                    save_path = os.path.join(flow_save_dir, '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                               os.path.basename(self.img_files[id2])))
                    np.save(save_path, pred_optical_flows[i])
                    gt_flow = np.load(os.path.join(self.seq_dir, 'raft_exhaustive',
                                                   '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                      os.path.basename(self.img_files[id2]))
                                                   ))
                    flow_error = np.linalg.norm(gt_flow - pred_optical_flows[i], axis=-1).mean()
                    flow_errors.append(flow_error)

                flow_errors = np.array(flow_errors)
                np.savetxt(os.path.join(self.out_dir, 'flow_error.txt'), flow_errors)

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'deform_mlp': de_parallel(self.deform_mlp).state_dict(),
                   'feature_mlp': de_parallel(self.feature_mlp).state_dict(),
                   'color_mlp': de_parallel(self.color_mlp).state_dict(),
                   'num_imgs': self.num_imgs
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.deform_mlp.load_state_dict(to_load['deform_mlp'])
        self.feature_mlp.load_state_dict(to_load['feature_mlp'])
        self.color_mlp.load_state_dict(to_load['color_mlp'])
        self.num_imgs = to_load['num_imgs']

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step
