import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Union, TypeVar, Tuple, Optional, Callable
T = TypeVar('T')

def convnxn(in_planes: int, out_planes: int, kernel_size: Union[T, Tuple[T]], stride: int = 1,
            groups: int = 1, dilation=1) -> nn.Conv1d:
    """nxn convolution and input size equals output size
    O = (I-K+2*P) / S + 1
    """
    if stride == 1:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 1) / 2)  # s = 1, to meet output size equals input size
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size,
                         dilation=dilation,
                         groups=groups, bias=False)
    elif stride == 2:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 2) / 2)  # s = 2, to meet output size equals input size // 2
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size,
                         dilation=dilation,
                         groups=groups, bias=False)
    else:
        raise Exception('No such stride, please select only 1 or 2 for stride value.')


class FFTLayer(nn.Module):

    def __init__(
        self,
        in_planes : int,
        length: int,
        wfb_switch: bool = False,
        filter_nums: int = 3
    ):
        super(FFTLayer, self).__init__()
        self.wfb_switch = wfb_switch
        self.filter_nums = filter_nums
        if not self.wfb_switch:
            self.parameter = nn.Parameter(torch.randn(size=(in_planes, int(length // 2) + 1, 2), dtype=torch.float32) * 1.0, requires_grad=True)
        else:
            self.parameter = nn.Parameter(torch.randn(size=(filter_nums, in_planes, int(length // 2) + 1, 2), dtype=torch.float32) * 1.0, requires_grad=True)
            self.GroupConv_R = GroupConv(in_channels=in_planes, filter_nums=filter_nums)
            self.GroupConv_I = GroupConv(in_channels=in_planes, filter_nums=filter_nums)
            self.conv_R = convnxn(in_planes=in_planes*filter_nums, out_planes=in_planes, kernel_size=3)
            self.conv_I = convnxn(in_planes=in_planes*filter_nums, out_planes=in_planes, kernel_size=3)


    def forward_fft(self, x: torch.Tensor) -> torch.Tensor:
        _, _, L = x.shape
        fft_x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.parameter)
        fft_x = fft_x * weight
        x_star = torch.fft.irfft(fft_x, dim=-1, n=L, norm='ortho')
        return x_star
    
    def forward_wfb_fft(self, x: torch.Tensor) -> torch.Tensor:
        _, _, L = x.shape
        fft_x = torch.fft.rfft(x, dim=-1, norm='ortho')
        W_R, W_I = self.parameter[:, :, :, 0], self.parameter[:, :, :, 1]
        X_R, X_I = fft_x.real, fft_x.imag
        C_R = self.GroupConv_R(X_R, W_R)
        C_I = self.GroupConv_I(X_I, W_I)

        Kf_R = self.conv_R(C_R)
        Kf_I = self.conv_I(C_I)
        Kf = torch.complex(Kf_R, Kf_I)

        fft_x = fft_x * Kf
        x_star = torch.fft.irfft(fft_x, dim=-1, n=L, norm='ortho')
        return x_star
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.wfb_switch:
            return self.forward_fft(x)
        return self.forward_wfb_fft(x)
 

class GroupConv(nn.Module):

    def __init__(self, in_channels, filter_nums=3, ratio=2, kernel_size=3):
        super().__init__()
        self.filter_nums = filter_nums
        self.conv1 = convnxn(in_planes=in_channels, out_planes=in_channels * filter_nums * ratio, kernel_size=kernel_size, groups=1)
        self.act = nn.SELU(inplace=True)
        self.conv2 = convnxn(in_planes=in_channels * filter_nums * ratio, out_planes=in_channels * filter_nums, kernel_size=kernel_size, groups=filter_nums)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, w):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        _, C, _ = x.shape
        x = torch.split(x, int(C / self.filter_nums), dim=1)
        x = [i for i in x]
        x = torch.cat([x[i] * w[i] for i in range(self.filter_nums)], dim=1)
        return x


if __name__ == '__main__':
    x = torch.randn(size=(10, 10, 200))
    plt.plot(x[0][0])
    model = FFTLayer(in_planes=10, length=200, wfb_switch=True, filter_nums=3)
    res = model(x)
    plt.plot(res[0][0].detach().numpy())
    plt.show()
    print(res[0][0].detach().numpy())