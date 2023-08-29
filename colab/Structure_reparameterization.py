import os
from collections import OrderedDict
import torch
import torch.nn as nn
from u2net.src.rep_half_u2net import rep_u2net_lite_half_m48 as rep_h_u2net_48
from src.model import u2net_lite_half_m48

#转换的模型转换前参数长度458 转换后长度应该为162(共74个convbn需要重构，重构前6个参数列，重构后2个参数列，其余还有sidemodule有6个conv和6个偏置，outconv一个conv一个偏置）
# 将传入的巻积参数进行运算，返回新巻积层的巻积核（权重）和偏置
def count_new_conv(conv,bn_bias,bn_weight,bn_running_mean,bn_running_var,eps):
    cnt = OrderedDict()
    for k,v in conv.items():
        list_name = k.split('.')
        if len(list_name) == 5:
            b = ".".join(list_name[:3])
            k_bias,k_weight,k_running_mean,k_running_var = b+'.bn.bias',b+'.bn.weight',b+'.bn.running_mean',b+'.bn.running_var'
        if len(list_name) == 6:
            b = ".".join(list_name[:4])
            k_bias, k_weight, k_running_mean, k_running_var = b + '.bn.bias', b + '.bn.weight', b + '.bn.running_mean', b + '.bn.running_var'
        std = (bn_running_var[k_running_var] + eps).sqrt()
        t = (bn_weight[k_weight]/std).reshape(-1,1,1,1)
        kernel = v * t
        bias = bn_bias[k_bias] - bn_running_mean[k_running_mean] * bn_weight[k_weight] / std
        key_weight = b+'.conv.weight'
        key_bias = b+'.conv.bias'
        cnt[key_weight] = kernel
        cnt[key_bias] = bias
    return cnt
def rep_weights():
    save_weights = 'save_weights/'
    mode = 'rep_h_u2net_48'
    torch.random.manual_seed(0)

    f1 = torch.randn(1, 2, 3, 3)

    new_module = rep_h_u2net_48()
    new_module.eval()

    pretrained_dict = torch.load('save_weights/u2net_lite_half_48m_best/model_best.pth')

    #new_dict = new_module.state_dict()
    old_module = u2net_lite_half_m48()
    old_module.eval()
    h = pretrained_dict['model']
    #加载原模型参数
    # 4. 遍历模型和预训练字典
    bn = nn.BatchNorm2d(num_features=2)
    eps = bn.eps
    every_mul_conv = OrderedDict()
    conv = {}
    bn_weight = {}
    bn_bias = {}
    bn_running_mean = {}
    bn_running_var = {}
    for name in h:
        #将name名分开，以便辨别是encode还是decode
        list_name = name.split('.')
        if list_name[0]=='encode_modules' or list_name[0]=='decode_modules':
            if 'conv.weight' in name:
                conv[name] = h[name]
            if 'bn.weight' in name:
                bn_weight[name] = h[name]
            if 'bn.bias' in name:
                bn_bias[name] = h[name]
            if 'bn.running_mean' in name:
                bn_running_mean[name] = h[name]
            if 'bn.running_var' in name:
                bn_running_var[name] = h[name]
            if len(conv) == 74 and len(bn_bias) == 74 and len(bn_weight)== 74 and len(bn_running_var) == 74 and len(bn_running_mean) == 74:
                new_weight = count_new_conv(conv,bn_bias,bn_weight,bn_running_mean,bn_running_var,eps)
                for k,v in new_weight.items():
                    every_mul_conv[k] = v
        if list_name[0]=='side_modules' or list_name[0]=='out_conv':
            every_mul_conv[name] = h[name]

    save_file = {"model": every_mul_conv,
                 "optimizer": pretrained_dict['optimizer'],
                 "lr_scheduler": pretrained_dict['lr_scheduler'],
                 "epoch": pretrained_dict['epoch'],
                 "args": pretrained_dict['args']}
    if not os.path.exists(save_weights+mode+"/model_best.pth"):
        os.makedirs(save_weights+mode,exist_ok=True)
    torch.save(save_file, save_weights+mode+"/model_best.pth")


    print("convert module has been tested, and the result looks good!")

def load_weights(root,path):
    f = torch.ones((1, 1, 4, 4))
    h = torch.load(root)
    m = torch.load(path)
    new_module = rep_h_u2net_48()
    old_module = u2net_lite_half_m48()
    old_module.eval()
    new_module.eval()
    if "model" in h:
        new_module.load_state_dict(h["model"])
    else:
        new_module.load_state_dict(h)
    old_module.load_state_dict(m["model"])
    f1 = new_module(f)
    f2 = old_module(f)

    return f1,f2


if __name__ == '__main__':
    rep_weights()
    root = 'save_weights/rep_h_u2net_48/model_best.pth'
    path = 'save_weights/u2net_lite_half_m48/model_best.pth'
    #h = load_weights(root,path)