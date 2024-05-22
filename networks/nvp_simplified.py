import numpy as np
import torch
from torch import masked_select, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import networks.pe_relu


class CouplingLayer(nn.Module):
    def __init__(self, map_st, projection, mask):
        super().__init__()
        self.map_st = map_st
        self.projection = projection
        # self.mask = mask
        self.register_buffer("mask", mask) ## register_buffer to save in state_dict

    def forward(self, F, y):
        y1 = y * self.mask
        F_y1 = torch.cat([F, self.projection(y[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_y1)
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.clamp(s, min=-8, max=8)
        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x[..., self.mask.squeeze().bool()])], dim=-1)
        st = self.map_st(F_x1)
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.clamp(s, min=-8, max=8)
        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class MLP(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Linear(d_in, d_out))
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Linear(d_in, c_out))
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out

    def forward(self, x):
        # x: B,...,C_in
        input_shape = x.shape
        C = input_shape[-1]
        _x = x.reshape(-1, C)  # X, C_in
        y = self.mlp(_x)  # X, C_out
        y = y.reshape(*input_shape[:-1], self.c_out)
        return y


def apply_homography_xy1(mat, xy1):
    """
    :param mat (*, 3, 3) (# * dims must match uv dims)
    :param xy1 (*, H, W, 3)
    :returns warped coordinates (*, H, W, 2)
    """
    out_h = torch.matmul(mat, xy1[..., None])
    return out_h[..., :2, 0] / (out_h[..., 2:, 0] + 1e-8)


def apply_homography(mat, uv):
    """
    :param mat (*, 3, 3) (# * dims must match uv dims)
    :param uv (*, H, W, 2)
    :returns warped coordinates (*, H, W, 2)
    """
    uv_h = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)  # (..., 3)
    return apply_homography_xy1(mat, uv_h)


class NVPSimplified(nn.Module):
    def __init__(
            self,
            n_layers,
            feature_dims,
            hidden_size,
            proj_dims,
            code_proj_hidden_size=[],
            proj_type="simple",
            pe_freq=4,
            normalization=True,
            affine=False,
            activation=nn.LeakyReLU,
            device='cuda',
    ):
        super().__init__()
        self._checkpoint = False
        self.affine = affine

        # make layers
        input_dims = 3
        normalization = nn.BatchNorm1d if normalization else None

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.code_projectors = nn.ModuleList()
        self.layer_idx = [i for i in range(n_layers)]

        i = 0
        mask_selection = []
        while i < n_layers:
            mask_selection.append(torch.randperm(input_dims))
            i += input_dims
        mask_selection = torch.cat(mask_selection)

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        for i in self.layer_idx:
            # get mask
            mask2 = torch.zeros(input_dims, device=device)
            mask2[mask_selection[i]] = 1
            mask1 = 1 - mask2

            # get transformation
            map_st = nn.Sequential(
                MLP(
                    proj_dims + feature_dims,
                    2,
                    hidden_size,
                    bn=normalization,
                    act=activation,
                    )
            )

            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type, pe_freq=pe_freq)
            self.layers1.append(CouplingLayer(map_st, proj, mask1[None, None, None]))

            # get code projector
            if len(code_proj_hidden_size) == 0:
                code_proj_hidden_size = [feature_dims]
            self.code_projectors.append(
                MLP(
                    feature_dims,
                    feature_dims,
                    code_proj_hidden_size,
                    bn=normalization,
                    act=activation,
                )
            )

        if self.affine:
            # this mlp takes time and depth as input and produce an affine transformation for x and y
            self.affine_mlp = networks.pe_relu.MLP(input_dim=2,
                                                   hidden_size=256,
                                                   n_layers=2,
                                                   skip_layers=[],
                                                   use_pe=True,
                                                   pe_dims=[1],
                                                   pe_freq=pe_freq,
                                                   output_dim=5).to(device)

    def _expand_features(self, F, x):
        _, N, K, _ = x.shape
        return F[:, None, None, :].expand(-1, N, K, -1)

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def invert_affine(self, a, b, c, d, tx, ty, zeros, ones):
        determinant = a * d - b * c

        inverse_determinant = 1.0 / determinant

        inverted_a = d * inverse_determinant
        inverted_b = -b * inverse_determinant
        inverted_c = -c * inverse_determinant
        inverted_d = a * inverse_determinant
        inverted_tx = (b * ty - d * tx) * inverse_determinant
        inverted_ty = (c * tx - a * ty) * inverse_determinant

        return torch.cat([inverted_a, inverted_b, inverted_tx,
                          inverted_c, inverted_d, inverted_ty,
                          zeros, zeros, ones], dim=-1).reshape(*a.shape[:-1], 3, 3)

    def get_affine(self, theta, inverse=False):
        """
        expands the 5 parameters into 3x3 affine transformation matrix
        :param theta (..., 5)
        :returns mat (..., 3, 3)
        """
        angle = theta[..., 0:1]
        scale1 = torch.exp(theta[..., 1:2])
        scale2 = torch.exp(theta[..., 3:4])
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        a = cos * scale1
        b = -sin * scale1
        c = sin * scale2
        d = cos * scale2
        tx = theta[..., 2:3]
        ty = theta[..., 4:5]
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        if inverse:
            return self.invert_affine(a, b, c, d, tx, ty, zeros, ones)
        else:
            return torch.cat([a, b, tx, c, d, ty, zeros, zeros, ones], dim=-1).reshape(*theta.shape[:-1], 3, 3)

    def _affine_input(self, t, x, inverse=False):
        depth = x[..., -1]  # [n_imgs, n_pts, n_samples]
        net_in = torch.stack([t[..., None].repeat(1, *x.shape[1:3]), depth], dim=-1)
        affine = self.get_affine(self.affine_mlp(net_in), inverse=inverse)  # [n_imgs, n_pts, n_samples, 3, 3]
        xy = x[..., :2]
        xy = apply_homography(affine, xy)
        x = torch.cat([xy, depth[..., None]], dim=-1)
        return x

    def forward(self, t, feat, x):
        y = x
        if self.affine:
            y = self._affine_input(t, y)
        for i in self.layer_idx:
            feat_i = self.code_projectors[i](feat)
            feat_i = self._expand_features(feat_i, y)
            l1 = self.layers1[i]
            y, _ = self._call(l1, feat_i, y)
        return y

    def inverse(self, t, feat, y):
        x = y
        for i in reversed(self.layer_idx):
            feat_i = self.code_projectors[i](feat)
            feat_i = self._expand_features(feat_i, x)
            l1 = self.layers1[i]
            x, _ = self._call(l1.inverse, feat_i, x)
        if self.affine:
            x = self._affine_input(t, x, inverse=True)
        return x


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dims):
        super().__init__()
        self._input_dims = input_dims

    @property
    def proj_dims(self):
        return self._input_dims

    def forward(self, x):
        return x


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2 * proj_dims), nn.ReLU(), nn.Linear(2 * proj_dims, proj_dims)
        )

    @property
    def proj_dims(self):
        return self._proj_dims

    def forward(self, x):
        return self.proj(x)


class FixedPositionalEncoding(ProjectionLayer):
    def __init__(self, input_dims, frequency, proj_dims):
        super().__init__(input_dims, proj_dims)
        ll = frequency
        self.sigma = np.pi * torch.pow(2, torch.linspace(0, ll - 1, ll, device='cuda')).view(1, -1)
        self.proj = nn.Sequential(
            nn.Linear(input_dims + input_dims * ll * 2, proj_dims), nn.LeakyReLU()
        )

    @property
    def proj_dims(self):
        return self._proj_dims * 3

    def forward(self, x):
        encoded = torch.cat(
            [
                torch.sin(x[:, :, :, :, None] * self.sigma[None, None, None]),
                torch.cos(x[:, :, :, :, None] * self.sigma[None, None, None]),
            ],
            dim=-1,
        ).view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = torch.cat([x, encoded], dim=-1)
        return self.proj(x)


class GaussianRandomFourierFeatures(ProjectionLayer):
    def __init__(self, input_dims, proj_dims, gamma=1.0):
        super().__init__(input_dims, proj_dims)
        self._two_pi = 2 * np.pi
        self._gamma = gamma
        ll = proj_dims // 2
        self.register_buffer("B", torch.randn(3, ll))

    def forward(self, x):
        xB = x.matmul(self.B * self._two_pi * self._gamma)
        return torch.cat([torch.cos(xB), torch.sin(xB)], dim=-1)


class GaborLayer(nn.Module):
    def __init__(self, input_dims, proj_dims, alpha=1., beta=1.0, weight_scale=128):
        super().__init__()
        self.linear = nn.Linear(input_dims, proj_dims)
        self.mu = nn.Parameter(2 * torch.rand(proj_dims, input_dims) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((proj_dims,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.linear2 = nn.Linear(input_dims, proj_dims)
        self.linear2.weight.data.uniform_(
            -np.sqrt(weight_scale / proj_dims),
            np.sqrt(weight_scale / proj_dims)
        )

    def forward(self, x):
        D = (
                (x ** 2).sum(-1)[..., None]
                + (self.mu ** 2).sum(-1)[None, :]
                - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :]) * self.linear2(x)


def get_projection_layer(**kwargs):
    type = kwargs["type"]

    if type == "identity":
        return IdentityProjection(3)
    elif type == "simple":
        return ProjectionLayer(2, kwargs.get("proj_dims", 128))
    elif type == "fixed_positional_encoding":
        return FixedPositionalEncoding(2, kwargs.get("pe_freq", 4), kwargs.get("proj_dims", 128))
    elif type == "gaussianrff":
        return GaussianRandomFourierFeatures(
            3, kwargs.get("proj_dims", 10), kwargs.get("gamma", 1.0)
        )
    elif type == 'gabor':
        return GaborLayer(3, kwargs.get("proj_dims", 128))
