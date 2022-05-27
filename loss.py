from math import exp
import torch
import torch.nn.functional as F
import torch.nn as nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1, img2, val_range, window_size=11, window=None, size_average=True, full=False
):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret


class SSIL_Loss(nn.Module):
    """
    Scale and Shift Invariant Loss
    """

    def __init__(self, valid_threshold=1e-8, max_threshold=1e8):
        super(SSIL_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold

    def inverse(self, mat):
        bt, _, _ = mat.shape
        mat += 1e-7 * torch.eye(2, dtype=mat.dtype, device=mat.device)
        a = mat[:, 0, 0]
        b = mat[:, 0, 1]
        c = mat[:, 1, 0]
        d = mat[:, 1, 1]
        ad_bc = a * d - b * c
        out = torch.zeros_like(mat)
        mat = mat / ad_bc[:, None, None]
        out[:, 0, 0] = mat[:, 1, 1]
        out[:, 0, 1] = -mat[:, 0, 1]
        out[:, 1, 0] = -mat[:, 1, 0]
        out[:, 1, 1] = mat[:, 0, 0]
        return out

    def scale_pred_depth_mask(self, pred, gt, logger=None):
        b, c, h, w = pred.shape
        mask = (gt > self.valid_threshold) & (gt < self.max_threshold)  # [b, c, h, w]
        mask_float = mask.to(dtype=pred.dtype, device=pred.device)
        pred_valid = pred * mask_float  # [b, c, h, w]
        ones_valid_shape = torch.ones_like(pred_valid) * mask_float  # [b, c, h, w]
        pred_valid_ones = torch.cat(
            (pred_valid, ones_valid_shape), dim=1
        )  # [b, c+1, h, w]
        pred_valid_ones_reshape = pred_valid_ones.reshape(
            (b, c + 1, -1)
        )  # [b, c+1, h*w]

        A = torch.matmul(
            pred_valid_ones_reshape, pred_valid_ones_reshape.permute(0, 2, 1)
        )  # [b, 2, 2]

        # print(A)
        # A_inverse = (A + 1e-7 * torch.eye(2, dtype=A.dtype, device=A.device)).inverse() # this may get identity matrix in some versions of Pytorch. If it occurs, add 'print(A)' before it can solve it
        A_inverse = self.inverse(A)
        gt_valid = gt * mask_float
        gt_reshape = gt_valid.reshape((b, c, -1))  # [b, c, h*w]
        B = torch.matmul(
            pred_valid_ones_reshape, gt_reshape.permute(0, 2, 1)
        )  # [b, 2, 1]
        scale_shift = torch.matmul(A_inverse, B)  # [b, 2, 1]
        ones = torch.ones_like(pred)
        pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
        pred_scale_shift = torch.matmul(
            pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2), scale_shift
        )  # [b, h*w, 1]
        pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))

        return pred_scale_shift, mask

    def forward(self, pred, gt, logger=None):
        pred_scale, mask_valid = self.scale_pred_depth_mask(pred, gt)
        valid_pixs = torch.sum(mask_valid, (1, 2, 3))
        valid_batch = valid_pixs > 50
        diff = torch.abs(
            pred_scale * valid_batch[:, None, None, None] * mask_valid
            - gt * valid_batch[:, None, None, None] * mask_valid
        )
        loss = torch.sum(diff) / (
            torch.sum(valid_batch[:, None, None, None] * mask_valid) + 1e-8
        )
        return loss


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = target > 0
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )

        return loss
