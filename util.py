import numpy as np
import os, sys, time
import imageio
import cv2
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import socket
import contextlib
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
import subprocess


TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

torch.manual_seed(1234)
np.random.seed(0)

sigma2alpha = lambda sigma: 1. - torch.exp(-sigma)


def float2uint8(x):
    return (255. * x).astype(np.uint8)


def uint82float(img):
    return np.ascontiguousarray(img) / 255.


def skew(x):
    if 'torch' in str(x.dtype):
        return torch.tensor([[0, -x[2], x[1]],
                             [x[2], 0, -x[0]],
                             [-x[1], x[0], 0]],
                            device=x.device)
    else:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])


def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def homogenize(coord):
    coord = torch.cat((coord, torch.ones_like(coord[..., [0]])), -1)
    return coord


def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
    else:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.


def denormalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords * torch.tensor([w-1., h-1.], device=coords.device) / 2.
    else:
        return (coords + 1.) * torch.tensor([w-1., h-1.], device=coords.device) / 2.


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


def gen_grid_np(h, w, normalize=False, homogeneous=False):
    if normalize:
        lin_y = np.linspace(-1., 1., num=h)
        lin_x = np.linspace(-1., 1., num=w)
    else:
        lin_y = np.arange(0, h)
        lin_x = np.arange(0, w)
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)
    grid = np.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = np.concatenate([grid, np.ones_like(grid[..., :1])], axis=-1)
    return grid  # [h, w, 2 or 3]


def save_current_code(outdir):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = '.'
    dst_dir = os.path.join(outdir, 'code', '{}'.format(date_time))
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns(
                        'data*', 'OLD*',
                        'logs*', 'out*', 'runs*', '*.png', '*.mp4', '*__pycache__*',
                        '*.git*', '*.idea*', '*.zip', '*.jpg'))


def drawMatches(img1, img2, kp1, kp2, num_vis=200, idx_vis=None, radius=2, mask=None):
    num_pts = len(kp1)
    if idx_vis is None:
        if num_vis < num_pts:
            idx_vis = np.random.choice(num_pts, num_vis, replace=False)
        else:
            idx_vis = np.arange(num_pts)

    kp1_vis = kp1[idx_vis]
    kp2_vis = kp2[idx_vis]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1 = float2uint8(img1)
    img2 = float2uint8(img2)

    center = np.median(kp1, axis=0)

    set_max = range(128)
    colors = {m: i for i, m in enumerate(set_max)}
    colors = {m: (255 * np.array(plt.cm.hsv(i/float(len(colors))))[:3][::-1]).astype(np.int32)
              for m, i in colors.items()}

    if mask is not None:
        ind = np.argsort(mask)[::-1]
        kp1_vis = kp1_vis[ind]
        kp2_vis = kp2_vis[ind]
        mask = mask[ind]

    for i, (pt1, pt2) in enumerate(zip(kp1_vis, kp2_vis)):
        # random_color = tuple(np.random.randint(low=0, high=255, size=(3,)).tolist())
        coord_angle = np.arctan2(pt1[1] - center[1], pt1[0] - center[0])
        corr_color = np.int32(64 * coord_angle / np.pi) % 128
        color = tuple(colors[corr_color].tolist())

        if (pt1[0] <= w1 - 1) and (pt1[0] >= 0) and (pt1[1] <= h1 - 1) and (pt1[1] >= 0):
            img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), radius, color, -1, cv2.LINE_AA)
        if (pt2[0] <= w2 - 1) and (pt2[0] >= 0) and (pt2[1] <= h2 - 1) and (pt2[1] >= 0):
            if mask is not None and mask[i]:
                img2 = cv2.drawMarker(img2, (int(pt2[0]), int(pt2[1])), color, markerType=cv2.MARKER_CROSS,
                                      markerSize=int(5*radius), thickness=int(radius/2), line_type=cv2.LINE_AA)
            else:
                img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), radius, color, -1, cv2.LINE_AA)

    out = np.concatenate([img1, img2], axis=1)
    return out


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new


# tensor
def colorize(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3 or flow_uv.ndim == 4, 'input flow must have three or four dimensions'
    assert flow_uv.shape[-1] == 2, 'input flow must have shape [..., H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    if flow_uv.ndim == 4:
        return np.stack([flow_uv_to_colors(u_, v_, convert_to_bgr) for (u_, v_) in zip(u, v)], axis=0)
    else:
        return flow_uv_to_colors(u, v, convert_to_bgr)
