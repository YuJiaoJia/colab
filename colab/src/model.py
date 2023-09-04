from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1,alpha: float = 0.01):
        super().__init__()#super函数可以初始化父类的参数，以便子类可以继承父类的参数

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)#使用官方权重需要将偏置设置为true
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):#该类继承于ConvBNRelu类
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):#这里在init中的参数会传递给父类
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):#flag表示是否启用
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)#利用双线性插值将x1（上一个模块的输出）插值到encoder阶段的tensor
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


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


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
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
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape#输入图像时图像的原始高宽

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.cat([x, x2], dim=1))
            decode_outputs.insert(0, x)

        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)#将图像还原回收集时图像的高宽
            side_outputs.insert(0, x)#将x从头插入

        x = self.out_conv(torch.cat(side_outputs, dim=1))#out_conv是1*1卷积层

        if self.training:#非训练模式就返回sigmoid（）
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs #返回的是融合输出
        else:
            return torch.sigmoid(x)

##半u2net的改进
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

## 该算法是针对半u2net的多尺度连接的改进
class U2Net_half_change(nn.Module):
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
        encode_output = []
        for i, m in enumerate(self.encode_modules):
            n = self.side_modules[i]
            if i ==0:
                x = m(x)
                encode_output.append(n(x))
                encode_outputs.append(x)
            else:
                x = m(encode_outputs[i-1])
                x2 = F.interpolate(x, size=[h,w], mode='bilinear', align_corners=False)
                encode_output.append(n(x2))
                x = torch.cat((x,encode_outputs[i-1]),dim=1)
                if i != self.encode_num - 1:
                    x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
                encode_outputs.append(x)

        e = torch.cat(encode_output,dim=1)
        for m in self.decode_modules:
            res = m(e)
        res = self.out_conv(res)#out_conv是1*1卷积层

        if self.training:#非训练模式就返回sigmoid（）
            # do not use torch.sigmoid for amp safe
            return [res] + encode_output #返回的是融合输出
        else:
            return torch.sigmoid(res)

class U2Net_half_multiple(nn.Module):#最后是将1、1+2、1+2+3...的输出图像拼接然后通过解码器解码得到
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []
        i = 0
        out = 0
        for c in cfg["encode"]:
            i = i + 1
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                if i == 1:
                    side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
                    out = c[3]
                else:
                    side_list.append(nn.Conv2d(out + c[3], out_ch, kernel_size=3, padding=1))
                    out = c[3]
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
        encode_output = []
        #t_start = time_synchronized()
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            n = self.side_modules[i]
            if i == 0:
                x2 = m(x)
                x2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)
                encode_output.append(x2)
                encode_outputs.append(n(x2))
                if i != self.encode_num - 1:
                    x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            else:
                x = m(encode_output[i-1])
                x2 = F.interpolate(x, size=encode_output[i-1].shape[2:], mode='bilinear', align_corners=False)
                encode_output.append(x2)
                x2 = torch.cat((x2, encode_output[i-1]), dim=1)
                #x3 = F.interpolate(x2, size=[h, w], mode='bilinear', align_corners=False)
                encode_outputs.append(n(x2))
                if i != self.encode_num - 1:
                    x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = torch.cat(encode_outputs, dim=1)
        for m in self.decode_modules:
            x = m(x)

        x = self.out_conv(x)#out_conv是1*1卷积层

        if self.training:#非训练模式就返回sigmoid（）
            # do not use torch.sigmoid for amp safe
            return [x] + encode_outputs #返回的是融合输出
        else:
            return torch.sigmoid(x)

class U2Net_half_multiple_change(nn.Module):
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
                last_list = nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1)
        self.decode_modules = nn.ModuleList(decode_list)

        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = last_list

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape  # 输入图像时图像的原始高宽

        # collect encode outputs
        multi = []
        side = []#用于传递给解码器的图像
        for i, m in enumerate(self.encode_modules):
            n = self.side_modules[i]
            if i <= 1:
                x = m(x)
                mx = n(x)
                side.append(mx)
                if i ==1:
                    mx = torch.cat((mx,multi[i-1]),dim=1)
                multi.append(mx)
            else:
                x = torch.cat((x,multi[i-2]),dim=1)
                x = m(x)
                mx = n(x)
                side.append(mx)
                mx = torch.cat((mx, multi[i - 1]), dim=1)
                multi.append(mx)
                # if i != self.encode_num - 1:
                #     x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = torch.cat(side, dim=1)
        for m in self.decode_modules:
            x = m(x)

        x = self.out_conv(x)  # out_conv是1*1卷积层

        if self.training:  # 非训练模式就返回sigmoid（）
            # do not use torch.sigmoid for amp safe
            return [x] + side  # 返回的是融合输出
        else:
            return torch.sigmoid(x)

def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 32, 64, False, False],      # En1
                   [6, 64, 32, 128, False, False],    # En2
                   [5, 128, 64, 256, False, False],   # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],   # En5
                   [4, 512, 256, 512, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],   # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],    # De3
                   [6, 256, 32, 64, False, True],     # De2
                   [7, 128, 16, 64, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)


def u2net_lite(out_ch: int = 1):#轻量级net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }
    return U2Net(cfg, out_ch)

def u2net_litte(out_ch: int = 1):#轻轻量级net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[6, 1, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True]]  # De1
    }
    return U2Net(cfg, out_ch)

def u2net_lite_half_m32(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 32, 64, False, True],  # En1
                   [6, 64, 32, 64, False, True],  # En2
                   [5, 64, 32, 64, False, True],  # En3
                   [4, 64, 32, 64, False, True],  # En4
                   [4, 64, 32, 64, True, True],  # En5
                   [4, 64, 32, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 32, 64, True, True]]  # De1
    }
    return U2Net_half(cfg, out_ch)

def u2net_lite_half_m48(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 43, 64, False, True],  # En1
                   [6, 64, 43, 64, False, True],  # En2
                   [5, 64, 43, 64, False, True],  # En3
                   [4, 64, 43, 64, False, True],  # En4
                   [4, 64, 43, 64, True, True],  # En5
                   [4, 64, 43, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 43, 64, False, True]]  # De1
    }
    return U2Net_half(cfg, out_ch)

def u2net_lite_half_m64(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 64, 64, False, True],  # En1
                   [6, 64, 64, 64, False, True],  # En2
                   [5, 64, 64, 64, False, True],  # En3
                   [4, 64, 64, 64, False, True],  # En4
                   [4, 64, 64, 64, True, True],  # En5
                   [4, 64, 64, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 64, 64, False, True]]  # De1
    }
    return U2Net_half(cfg, out_ch)

def u2net_lite_half_m128(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 128, 64, False, True],  # En1
                   [6, 64, 128, 64, False, True],  # En2
                   [5, 64, 128, 64, False, True],  # En3
                   [4, 64, 128, 64, False, True],  # En4
                   [4, 64, 128, 64, True, True],  # En5
                   [4, 64, 128, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 128, 64, False, True]]  # De1
    }
    return U2Net_half(cfg, out_ch)

def u2net_lite_half_change_m32(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 32, 64, False, True],  # En1
                   [6, 64, 32, 32, False, True],  # En2
                   [5, 96, 32, 16, False, True],  # En3
                   [4, 112, 32, 16, False, True],  # En4
                   [4, 128, 32, 16, True, True],  # En5
                   [4, 144, 32, 16, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[6, 6, 32, 64, False, True]]  # De1
    }
    return U2Net_half_change(cfg, out_ch)

def U2Net_half_multiple_32m(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 32, 64, False, True],  # En1
                   [6, 64, 32, 64, False, True],  # En2
                   [5, 64, 32, 64, False, True],  # En3
                   [4, 64, 32, 64, False, True],  # En4
                   [4, 64, 32, 64, True, True],  # En5
                   [4, 64, 32, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 32, 64, False, True]]  # De1
    }
    return U2Net_half_multiple(cfg, out_ch)

def U2Net_half_multiple_change_32(out_ch: int = 1):#轻量级半u2net
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1, 32, 64, False, True],  # En1
                   [6, 64, 32, 64, False, True],  # En2
                   [5, 65, 32, 64, False, True],  # En3
                   [4, 66, 32, 64, False, True],  # En4
                   [4, 67, 32, 64, True, True],  # En5
                   [4, 68, 32, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 6, 32, 64, False, True]]  # De1
    }
    return U2Net_half_multiple_change(cfg, out_ch)

def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = u2net_full()
    convert_onnx(u2net, "u2net_full.onnx")
