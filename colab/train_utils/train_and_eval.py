import math
import torch
from torch.nn import functional as F
from . import distributed_utils as utils
import torch.nn as nn


import os
from collections import deque
def criterion(inputs, target):
    #input存储的是融合的预测特征图，由于类别只有前景和背景两类
    losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss

# 自定义 EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0,f1=None, mae=None, f1_tolerance=0.001, mae_tolerance=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.f1 = f1
        self.mae = mae
        self.f1_tolerance = f1_tolerance
        self.mae_tolerance = mae_tolerance
        self.f1_queue = deque(maxlen=10)
        self.mae_queue = deque(maxlen=10)

    def __call__(self, val_loss,save_weights,mode,save_file,f1,mae):
        # score = -val_loss
        # if self.best_score is None:
        #     self.best_score = score
        #     self.f1 = f1
        #     self.mae = mae
        #     self.save_checkpoint(val_loss,save_weights,mode,save_file)
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     if self.verbose:
        #         print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        # else:
        #     self.best_score = score
        #     self.f1 = f1
        #     self.mae = mae
        #     self.save_checkpoint(val_loss,save_weights,mode,save_file)
        #     self.counter = 0

        self.f1_queue.append(f1)
        self.mae_queue.append(mae)
        if len(self.f1_queue) > 10:
            self.f1_queue.pop(0)
        if len(self.mae_queue) > 10:
            self.mae_queue.pop(0)
        if len(self.f1_queue) == 10 and len(self.mae_queue) == 10:
            f1_diff = max(self.f1_queue) - min(self.f1_queue)
            mae_diff = max(self.mae_queue) - min(self.mae_queue)
            if f1_diff <= self.f1_tolerance and mae_diff <= self.mae_tolerance:
                self.early_stop = True
            else:
                self.save_checkpoint(val_loss, save_weights, mode, save_file)

    def save_checkpoint(self, val_loss,save_weights,mode,save_file):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(save_file, save_weights+mode+"/model_best.pth")
        self.val_loss_min = val_loss



#定义triplet损失函数
def triplet_loss(an, pn, fnn):
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # 计算 triplet 损失
    loss = triplet_loss(an, pn, fnn)
    return loss

#定义Quadruplet Loss损失函数
def Quadruplet_loss(an,pn,fnn1,fnn2):
    loss1 = triplet_loss(an,pn,fnn1)
    loss2 = triplet_loss(an,pn,fnn2)
    return loss1+loss2

def focal_loss(inputs, target, gamma=2.0, alpha=0.25):
    # 计算二元交叉熵损失函数
    bce_loss = criterion(inputs,target)
    pt = torch.exp(-bce_loss + 1e-5)#torch.exp函数计算pt，使得难分类的样本对损失函数的贡献更大。
    # 计算Focal Loss
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    # 求平均损失
    average_loss = torch.mean(focal_loss)
    return average_loss


def dice_loss(pred, target, smooth=1.):
    # pred = pred.contiguous()
    # target = target.contiguous()
    pred = sum(pred)

    intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=1)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2).sum(dim=1) + target.sum(dim=2).sum(dim=2).sum(dim=1) + smooth)))

    return loss.mean()
def bce_dice_loss(pred, target,bce_w=1.0,dice_w=1.0):
    bce_loss = criterion(pred,target)
    dice_loss_val = dice_loss(pred, target)
    return bce_w*bce_loss + dice_w*dice_loss_val

def evaluate(model, data_loader, device):
    model.eval()
    mae_metric = utils.MeanAbsoluteError()
    f1_metric = utils.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            # post norm
            # ma = torch.max(output)
            # mi = torch.min(output)
            # output = (output - mi) / (ma - mi)

            mae_metric.update(output, targets)
            f1_metric.update(output, targets)

        mae_metric.gather_from_all_processes()
        f1_metric.reduce_from_all_processes()
    return mae_metric, f1_metric


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        #更新学习率
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
