import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def cauchy_loss(pred, gt, c=1, mask=None, normalize=True):
    loss = torch.log(1 + ((pred - gt) / c)**2)
    if mask is not None:
        if normalize:
            return (loss * mask).mean() / (mask.mean() + 1e-8)
        else:
            return (loss * mask).mean()
    else:
        return loss.mean()


def masked_mse_loss(pred, gt, mask=None, normalize=True):
    if mask is None:
        return F.mse_loss(pred, gt)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction='none')
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile=1):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction='none').mean(dim=-1, keepdim=True)
        loss_at_quantile = torch.quantile(sum_loss, quantile)
        quantile_mask = (sum_loss < loss_at_quantile).squeeze(-1)
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (ndim * torch.sum(mask[quantile_mask]) + 1e-8)
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction='none')
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction='none').mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_std_normed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction='none')  # [..., d]
    mask = loss.mean(dim=-1) < torch.quantile(loss.mean(dim=-1), quantile)  # [...]
    pred_std = torch.std(pred[mask], dim=0)  # [d]
    gt_std = torch.std(gt[mask], dim=0)  # [d]
    std = 0.5 * (pred_std + gt_std)
    trimmed_std_normed_loss = (loss / std).mean()
    return trimmed_std_normed_loss


def trimmed_mse_loss(pred, gt, mask=None, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction='none').mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile]
    if mask is not None:
        mask = mask[loss < loss_at_quantile]
        loss = torch.mean(mask * trimmed_loss) / torch.mean(mask)
    else:
        loss = torch.mean(trimmed_loss)
    return loss


def trimmed_var_normed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction='none')  # [..., d]
    mask = loss.mean(dim=-1) < torch.quantile(loss.mean(dim=-1), quantile)  # [...]
    pred_var = torch.var(pred[mask], dim=0)  # [d]
    gt_var = torch.var(gt[mask], dim=0)  # [d]
    var = 0.5 * (pred_var + gt_var)
    trimmed_var_normed_loss = (loss / var).mean()
    return trimmed_var_normed_loss


def compute_depth_range_loss(depth, min_th=0, max_th=2):
    '''
    the depth of mapped 3d locations should also be within the near and far depth range
    '''
    loss_lower = ((depth[depth < min_th] - min_th)**2).sum() / depth.numel()
    loss_upper = ((depth[depth > max_th] - max_th)**2).sum() / depth.numel()
    return loss_upper + loss_lower


def lossfun_distortion(t, w):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return (loss_inter + loss_intra).mean()


def median_scale_shift(x):
    '''
    :param x: [batch, h, w]
    :return: median scaled and shifted x
    '''
    batch_size = len(x)
    median_x = torch.median(x.reshape(batch_size, -1), dim=1).values[:, None, None]
    s_x = torch.mean(torch.abs(x - median_x), dim=(1, 2), keepdim=True)
    return (x - median_x) / s_x


def scale_shift_invariant_loss(pred, gt):
    pred_ = median_scale_shift(pred)
    gt_ = median_scale_shift(gt)
    return torch.mean(torch.abs(pred_ - gt_))


def trimmed_scale_shift_invariant_loss(pred, gt, percentile=0.8):
    pred_ = median_scale_shift(pred)
    gt_ = median_scale_shift(gt)
    error = torch.abs(pred_ - gt_).flatten()
    cut_value = torch.quantile(error, percentile)
    return error[error < cut_value].mean()


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, model='vgg19', device='cuda'):
        super().__init__()
        if model == 'vgg16':
            self.vgg = Vgg16().to(device)
            self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]
        elif model == 'vgg19':
            self.vgg = Vgg19().to(device)
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            # self.weights = [1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5]
            # self.weights = [1/2.6, 1/4.8, 1/3.7, 1/5.6, 2/1.5]
        # self.criterion = nn.L1Loss()
        self.loss_func = masked_l1_loss

    @staticmethod
    def preprocess(x, size=224):
        # B, C, H, W
        min_in_size = min(x.shape[-2:])
        device = x.device
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        x = (x - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
        # if min_in_size <= size:
        #     mode = 'bilinear'
        #     align_corners = True
        # else:
        #     mode = 'area'
        #     align_corners = None
        # x = F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        return x

    def forward(self, x, y, mask=None, size=224):
        x = self.preprocess(x, size=size)    # assume x, y are inside (0, 1)
        y = self.preprocess(y, size=size)

        if mask is not None:
            if min(mask.shape[-2:]) <= size:
                mode = 'bilinear'
                align_corners = True
            else:
                mode = 'area'
                align_corners = None
            mask = F.interpolate(mask, size=size, mode=mode, align_corners=align_corners)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        loss = self.loss_func(x, y, mask)
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.loss_func(x_vgg[i], y_vgg[i], mask)
        return loss


def normalize_minus_one_to_one(x):
    x_min = x.min()
    x_max = x.max()
    return 2. * (x - x_min) / (x_max - x_min) - 1.


def get_flow_smoothness_loss(flow, alpha):
    flow_gradient_x = flow[:, :, :, 1:, :] - flow[:, :, :, -1:, :]
    flow_gradient_y = flow[:, :, :, :, 1:] - flow[:, :, :, :, -1:]
    cost_x = (alpha[:, :, :, 1:, :] * torch.norm(flow_gradient_x, dim=2, keepdim=True)).sum()
    cost_y = (alpha[:, :, :, :, 1:] * torch.norm(flow_gradient_y, dim=2, keepdim=True)).sum()
    avg_cost = (cost_x + cost_y) / (2 * alpha.sum() + 1e-6)
    return avg_cost

