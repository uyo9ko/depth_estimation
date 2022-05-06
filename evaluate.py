import numpy as np
import torch
from scipy.stats import pearsonr
import cv2

def basic_process_depth(gt_depth, pred_depth, pred_min_depth, pred_max_depth, gt_min_depth=0, gt_max_depth=40):
    pred_depth[pred_depth < pred_min_depth] = pred_min_depth
    pred_depth[pred_depth > pred_max_depth] = pred_max_depth
    pred_depth[torch.isinf(pred_depth)] = pred_max_depth
    pred_depth[torch.isnan(pred_depth)] = pred_min_depth
    valid_mask = torch.logical_and(gt_depth > gt_min_depth, gt_depth < gt_max_depth)
    gt_depth = gt_depth[valid_mask]
    pred_depth = pred_depth[valid_mask]
    return gt_depth, pred_depth

def process_depth(gt_depth, pred_depth, min_depth, max_depth):
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    pred_depth[torch.isinf(pred_depth)] = max_depth
    pred_depth[torch.isnan(pred_depth)] = min_depth
    valid_mask = torch.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    gt_depth = gt_depth[valid_mask]
    pred_depth = pred_depth[valid_mask]
    return gt_depth, pred_depth

def np_process_depth(gt_depth, pred_depth, min_depth, max_depth):
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    pred_depth[np.isinf(pred_depth)] = max_depth
    pred_depth[np.isnan(pred_depth)] = min_depth
    valid_mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    gt_depth = gt_depth[valid_mask]
    pred_depth = pred_depth[valid_mask]
    return gt_depth, pred_depth


def compute_acc(gt, pred):
    thresh = torch.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25).type(torch.float32).mean()


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25 ).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def display_metrics(metrics):
    num_samples = len(metrics)
    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    for i, metric in enumerate(metrics):
        silog[i] = metric[0]
        log10[i] = metric[1]
        rms[i] = metric[2]
        log_rms[i] = metric[3]
        abs_rel[i] = metric[4]
        sq_rel[i] = metric[5]
        d1[i] = metric[6]
        d2[i] = metric[7]
        d3[i] = metric[8]

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
    'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    metric_dict = {'d1': d1.mean(), 'd2': d2.mean(), 'd3': d3.mean(), 'abs_rel': abs_rel.mean(), 'rmse': rms.mean(), 'log10': log10.mean()}
    
    return metric_dict






def sq_sinv(y,y_):
    # To avoid log(0) = -inf
    y_[y_==0] = 1
    y[y==0] = 1
    alpha = np.mean(np.log(y_) - np.log(y))
    err = (np.log(y) - np.log(y_) + alpha) ** 2
    return (np.mean(err[:]) / 2)

def pear_coeff(y,y_):
    y = y.ravel()
    y_ = y_.ravel()
    err = pearsonr(y,y_)
    return err[0]

def compute_uw_errors(gt, pred):
    return sq_sinv(gt, pred),pear_coeff(gt, pred)


def display_uw_metrics(metrics):
    sq_sinv = np.zeros(len(metrics), np.float32)
    pear_coeff = np.zeros(len(metrics), np.float32)
    for i, metric in enumerate(metrics):
        sq_sinv[i] = metric[0]
        pear_coeff[i] = metric[1]
    print("{:>7}, {:>7}".format('SqSin', 'PearCoeff'))
    print("{:7.3f}, {:7.3f}".format(sq_sinv.mean(), pear_coeff.mean()))
    metric_dict = {'sq_sinv': sq_sinv.mean(), 'pear_coeff': pear_coeff.mean()}  
    return metric_dict


