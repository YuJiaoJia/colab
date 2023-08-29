from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()#super函数可以初始化父类的参数，以便子类可以继承父类的参数

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=True)#使用官方权重需要将偏置设置为true
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu((self.conv(x)))


class DownConvBNReLU(ConvBNReLU):#该类继承于ConvBNRelu类
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):#这里在init中的参数会传递给父类
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.conv(x))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):#flag表示是否启用
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)#利用双线性插值将x1（上一个模块的输出）插值到encoder阶段的tensor
        return self.relu(self.conv(torch.cat([x1, x2], dim=1)))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):#height表示深度
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]#第一个编码模块没有下采样
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]#第一个解码块，没有上采样过程
        for i in range(height - 2):#除去没有上下采样的模块，剩下的模块
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []#收集每个模块的输出
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()#将u最下面的模块输出弹出（即膨胀卷积输出）
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):#encoder5、encoder6、decoder5模块（将上下采样替换成了膨胀卷积）
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in

class U2Net_half(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            if c[5] is True:
                last_list=nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1)
        self.decode_modules = nn.ModuleList(decode_list)

        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = last_list

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape#输入图像时图像的原始高宽

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # 收集每层编码器对应的特征图（尺寸与原图像尺寸相同）
        side_outputs = []
        for m in self.side_modules:
            x = encode_outputs.pop()
            # 注意，这里不能直接把m函数加在inter前面，会因为程序异步性报错
            x = F.interpolate(m(x), size=[h,w], mode='bilinear', align_corners=False)#将图像还原回收集时图像的高宽
            side_outputs.insert(0, x)#将x从头插入

        x = torch.cat(side_outputs, dim=1)
        for m in self.decode_modules:
            x = m(x)

        x = self.out_conv(x)#out_conv是1*1卷积层

        if self.training:#非训练模式就返回sigmoid（）
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs #返回的是融合输出
        else:
            return torch.sigmoid(x)

def rep_u2net_lite_half_m48(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 48, 64, False, True],  # En1
                   [6, 64, 48, 64, False, True],  # En2
                   [5, 64, 48, 64, False, True],  # En3
                   [4, 64, 48, 64, False, True],  # En4
                   [4, 64, 48, 64, True, True],  # En5
                   [4, 64, 48, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 48, 64, False, True]]  # De1
    }
    return U2Net_half(cfg, out_ch)