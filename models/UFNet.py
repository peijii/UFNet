import torch
import torch.nn as nn
from fftlayer import FFTLayer
from dwtlayer import DWTLayer
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


# ===================== I2CBlock ===========================
class I2CBlockv1(nn.Module):

    def __init__(
            self,
            in_planes: int,
            expansion_rate: int = 1,
            intra_kernel_size: int = 3,
            inter_kernel_size: int = 1,
            stride: int = 1,
            groups: int = 10,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ac_flag: bool = True,
    ):
        """
        input size: [B, C, N]
        output size: [B, e*(C+1), N]
        """
        super(I2CBlockv1, self).__init__()
        self.C = in_planes
        self.intra_kernel_size = intra_kernel_size
        self.inter_kernel_size = inter_kernel_size
        self.groups = groups
        self.e = expansion_rate
        self.flag = ac_flag
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.group_width = int(self.C / self.groups)

        self.inter_conv = convnxn(self.C, self.group_width * self.e, kernel_size=self.inter_kernel_size, stride=stride,
                                  groups=1)
        self.intra_conv = convnxn(self.C, self.C * self.e, kernel_size=self.intra_kernel_size, stride=stride,
                                  groups=groups)
        if self.flag:
            self.bn = norm_layer((self.group_width + self.C) * self.e)
            self.act = nn.SELU(inplace=True)

        self._init_weights()

    def forward(self, x):

        inter_output = self.inter_conv(x)
        intra_output = self.intra_conv(x)
        out = torch.cat((intra_output, inter_output), 1)
        if self.flag:
            out = self.bn(out)
            out = self.act(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class I2CBlockv2(nn.Module):

    def __init__(
            self,
            in_planes: int,
            expansion_rate: Union[int, float] = 1,
            stride: int = 1,
            intra_kernel_size: int = 3,
            inter_kernel_size: int = 1,
            groups: int = 10 + 1,
            length: int = 100,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ac_flag: bool = True,
            fft: bool = False,
            dwt: bool = False
    ):
        """
        input size: [B, C, N]
        output size: [B, e*C, N]
        """
        super(I2CBlockv2, self).__init__()
        self.groups = groups
        self.e = expansion_rate
        self.intra_kernel_size = intra_kernel_size
        self.inter_kernel_size = inter_kernel_size
        self.ac_flag = ac_flag
        self.fft = fft
        self.dwt = dwt
        self.group_width = int(in_planes / self.groups)

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if self.fft and not self.dwt:
            self.inter_tsp1 = FFTLayer(dim=int(self.group_width * (self.groups - 1)), length=length)
            self.inter_tsp2 = FFTLayer(dim=int(self.group_width * (self.e + 1)), length=length)
            self.intra_tsp1 = FFTLayer(dim=int(self.group_width * (self.groups - 1)), length=length)

        elif not self.fft and self.dwt:
            self.inter_tsp1 = DWTLayer(levels=1)
            self.inter_tsp2 = DWTLayer(levels=1)
            self.intra_tsp1 = DWTLayer(levels=1)

        elif self.fft and self.dwt:
            self.inter_tsp1 = nn.Sequential(DWTLayer(levels=1), FFTLayer(dim=int(self.group_width * (self.groups - 1)), length=length))
            self.inter_tsp2 = nn.Sequential(DWTLayer(levels=1), FFTLayer(dim=int(self.group_width * (self.e + 1)), length=length))
            self.intra_tsp1 = nn.Sequential(DWTLayer(levels=1), FFTLayer(dim=int(self.group_width * (self.groups - 1)), length=length))
        else:
            pass

        self.inter_channel1 = convnxn(self.group_width * (self.groups - 1), int(self.group_width * self.e),
                                      kernel_size=self.inter_kernel_size, stride=stride, groups=1)

        self.inter_channel2 = convnxn(int(self.group_width * (self.e + 1)), int(self.group_width * self.e),
                                      kernel_size=self.inter_kernel_size, stride=stride, groups=1)

        self.intra_channel = convnxn(int(self.group_width * (self.groups - 1)),
                                     int((self.group_width * (self.groups - 1)) * self.e),
                                     kernel_size=self.intra_kernel_size, stride=stride, groups=self.groups - 1)

        if self.ac_flag:
            self.bn = norm_layer(int((self.group_width * (self.groups - 1)) * self.e + self.group_width * self.e))
            self.act = nn.SELU(inplace=True)

        self._init_weights()

    def forward_wo_s(self, x):
        intra_data_prev = x[:, :(self.groups - 1) * self.group_width, :]
        inter_data_prev = x[:, (self.groups - 1) * self.group_width:, :]

        inter_data_current1 = self.inter_channel1(intra_data_prev)
        inter_data_current2 = torch.cat((inter_data_prev, inter_data_current1), 1)
        inter_data_current = self.inter_channel2(inter_data_current2)

        intra_data_current = self.intra_channel(intra_data_prev)

        output = torch.cat((intra_data_current, inter_data_current), 1)

        if self.ac_flag:
            output = self.bn(output)
            output = self.act(output)

        return output

    def forward_w_s(self, x):
        intra_data_prev = x[:, :(self.groups - 1) * self.group_width, :]
        inter_data_prev = x[:, (self.groups - 1) * self.group_width:, :]

        intra1_fft_data = self.inter_tsp1(intra_data_prev)
        inter_data_current1 = self.inter_channel1(intra1_fft_data)

        inter_data_current2 = torch.cat((inter_data_prev, inter_data_current1), 1)
        inter_fft_data = self.inter_tsp2(inter_data_current2)
        inter_data_current = self.inter_channel2(inter_fft_data)

        intra2_fft_data = self.intra_tsp1(intra_data_prev)
        intra_data_current = self.intra_channel(intra2_fft_data)

        output = torch.cat((intra_data_current, inter_data_current), 1)

        if self.ac_flag:
            output = self.bn(output)
            output = self.act(output)

        return output

    def forward(self, x):
        if self.fft or self.dwt:
            return self.forward_w_s(x)
        else:
            return self.forward_wo_s(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ===================== I2CMSE Block =======================
class I2CMSE(nn.Module):

    def __init__(
            self,
            in_planes: int,
            groups: int = 11,
            b1_size: int = 5,
            b2_size: int = 11,
            b3_size: int = 21,
            expansion_rate: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            fft: bool = False,
            dwt: bool = False
    ) -> None:
        super(I2CMSE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.in_planes = in_planes
        self.groups = groups
        self.group_width = int(in_planes / groups)
        self.expansion = expansion_rate
        self.b1_size = b1_size
        self.b2_size = b2_size
        self.b3_size = b3_size

        self.branch1_1 = I2CBlockv2(in_planes=in_planes, expansion_rate=self.expansion, intra_kernel_size=self.b1_size, inter_kernel_size=1, stride=1, groups=self.groups, ac_flag=False, fft=fft, dwt=dwt)
        self.branch1_2 = I2CBlockv2(in_planes=in_planes*self.expansion, expansion_rate=1, intra_kernel_size=self.b1_size, inter_kernel_size=1, stride=1, groups=self.groups, ac_flag=True, fft=fft, dwt=dwt)

        self.branch2_1 = I2CBlockv2(in_planes=in_planes, expansion_rate=self.expansion, intra_kernel_size=self.b2_size, inter_kernel_size=1, stride=1, groups=self.groups, ac_flag=False, fft=fft, dwt=dwt)
        self.branch2_2 = I2CBlockv2(in_planes=in_planes*self.expansion, expansion_rate=1, intra_kernel_size=self.b2_size, inter_kernel_size=1, stride=1, groups=self.groups, ac_flag=True, fft=fft, dwt=dwt)

        self.branch3_1 = I2CBlockv2(in_planes=in_planes, expansion_rate=self.expansion, intra_kernel_size=self.b3_size, inter_kernel_size=1, stride=1, groups=self.groups, ac_flag=False, fft=fft, dwt=dwt)
        self.branch3_2 = I2CBlockv2(in_planes=in_planes*self.expansion, expansion_rate=1, intra_kernel_size=self.b3_size, inter_kernel_size=1, stride=1, groups=self.groups, ac_flag=True, fft=fft, dwt=dwt)

        self.shrinkage = convnxn(3*in_planes*self.expansion, in_planes*self.expansion, kernel_size=3, stride=1, groups=self.groups)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branch1 = self.branch1_1(x)
        branch1_out = self.branch1_2(branch1)

        branch2 = self.branch2_1(x)
        branch2_out = self.branch2_2(branch2)

        branch3 = self.branch3_1(x)
        branch3_out = self.branch3_2(branch3)

        outputs = [
            torch.cat([branch1_out[:,
                       int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :],
                       branch2_out[:,
                       int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :],
                       branch3_out[:,
                       int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :]],
                      1)
            for i in range(self.groups)]

        out = torch.cat(outputs, 1)
        out = self.shrinkage(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ===================== Rearrange ==========================
class Rearrange(nn.Module):

    def __init__(self, group_width, groups):
        super(Rearrange, self).__init__()
        self.group_width = group_width
        self.groups = groups
        
    def forward(self, dwt, fft, raw):
        outputs = [
            torch.cat([dwt[:,
                       int(i * self.group_width):int((i + 1) * self.group_width), :],
                       fft[:,
                       int(i * self.group_width):int((i + 1) * self.group_width), :],
                       raw[:,
                       int(i * self.group_width):int((i + 1) * self.group_width), :]],
                      1)
            for i in range(self.groups)]
        return torch.cat(outputs, 1)

# ===================== UFBlock ============================
class UFBlock(nn.Module):

    def __init__(
            self,
            in_planes: int,
            length: int,
            groups: int,
            expansion_rate: int,
            intra_kernel_size: int = 3,
            inter_kernel_size: int = 1
    ):
        super(UFBlock, self).__init__()
        self.in_planes = in_planes
        self.groups = groups
        self.group_width = int(in_planes / groups)

        self.intra_dwt_branch = DWTLayer(levels=1)
        self.intra_fft_branch = FFTLayer(dim=self.group_width*(self.groups-1), length=length)
        self.rearrange1 = Rearrange(group_width=self.group_width, groups=self.groups-1)

        # self.intra_intra_conv = nn.Conv1d(in_channels=in_planes, out_channels=expansion_rate*in_planes, kernel_size=intra_kernel_size, groups=groups)
        # self.intra_inter_conv = nn.Conv1d(in_channels=in_planes, out_channels=expansion_rate, kernel_size=inter_kernel_size, groups=1)
        
        self.inter_dwt_branch = DWTLayer(levels=1)
        self.inter_fft_branch = FFTLayer(dim=self.group_width*1, length=length)
        self.rearrange2 = Rearrange(group_width=self.group_width, groups=1)

        # self.inter_inter_conv = nn.Conv1d(in_channels=in_planes, out_channels=in_planes*expansion_rate, kernel_size=inter_kernel_size, groups=1)
        # self.calibration = nn.Sequential([
        #     nn.Conv1d(in_channels=in_planes, out_channels=expansion_rate, kernel_size=inter_kernel_size, groups=1),
        #     nn.BatchNorm1d(expansion_rate),
        #     nn.SELU(inplace=True)
        # ])

    def forward(self, x):
        intra_X = x[:, :self.group_width*(self.groups-1), :]
        print(intra_X.shape)
        inter_X = x[:, self.group_width*(self.groups-1):, :]
        print(inter_X.shape)

        intra_dwt_f = self.intra_dwt_branch(intra_X)
        intra_fft_f = self.intra_fft_branch(intra_X)
        # intra_branch_f = torch.cat([intra_dwt_f, intra_fft_f, intra_X], 1)

        outputs = self.rearrange1(intra_dwt_f, intra_fft_f, intra_X)


        # intra_branch_f = self.rearrange1(intra_branch_f)
        # Intra_output = self.intra_intra_conv(intra_branch_f)
        # intra_branch_inter_f = self.intra_inter_conv(intra_branch_f)

        # inter_dwt_f = self.inter_dwt_branch(inter_X)
        # inter_fft_f = self.inter_fft_branch(inter_X)
        # inter_branch_f = torch.concat([inter_dwt_f, inter_fft_f, inter_X], 1)
        # inter_branch_f = self.rearrange2(inter_branch_f)
        # inter_branch_inter_f = self.inter_inter_conv(inter_branch_f)
        # Inter_output = torch.concat([intra_branch_inter_f, inter_branch_inter_f])
        # Inter_output = self.calibration(Inter_output)

        # output = torch.concat([Intra_output, Inter_output], 1)

        return outputs


# =================== I2CNet ======================
class UFNet(nn.Module):

    def __init__(
            self,
            in_planes: int = 10,
            length: int = 100,
            num_classes: int = 52,
            mse_b1: int = 5,
            mse_b2: int = 11,
            mse_b3: int = 21,
            mse_expansions: list = [1, 1, 1],
            uf_expansions: list = [1, 1, 1],
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            fft_flag: bool = False,
            dwt_flag: bool = False
    ) -> None:
        super(UFNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.groups = in_planes
        self.mse_expansions = mse_expansions

        self.conv1 = I2CBlockv1(in_planes=in_planes, expansion_rate=1, intra_kernel_size=3, inter_kernel_size=1, groups=self.groups)

        self.in_planes = in_planes + 1
        self.groups += 1

        self.mse1 = I2CMSE(in_planes=self.in_planes, groups=self.groups, b1_size=mse_b1, b2_size=mse_b2, b3_size=mse_b3, expansion_rate=self.mse_expansions[0], fft=fft_flag, dwt=dwt_flag)
        self.mse2 = I2CMSE(in_planes=self.in_planes*self.mse_expansions[0], groups=self.groups, b1_size=mse_b1, b2_size=mse_b2, b3_size=mse_b3, expansion_rate=self.mse_expansions[1], fft=fft_flag, dwt=dwt_flag)
        self.mse3 = I2CMSE(in_planes=self.in_planes*self.mse_expansions[0]*self.mse_expansions[1], groups=self.groups, b1_size=mse_b1, b2_size=mse_b2, b3_size=mse_b3, expansion_rate=self.mse_expansions[2], fft=fft_flag, dwt=dwt_flag)

        self.mse1_out_planes = self.in_planes * self.mse_expansions[0]
        self.mse2_out_planes = self.in_planes * self.mse_expansions[0] * self.mse_expansions[1]
        self.mse3_out_planes = self.in_planes * self.mse_expansions[0] * self.mse_expansions[1] * self.mse_expansions[2]

        self.uf1 = UFBlock(in_planes=self.mse3_out_planes, length=length, groups=self.groups, expansion_rate=1)
        # self.uf2 = UFBlock()
        # self.uf3 = UFBlock()
        
        # self.adaptiveAvgPool1d = nn.AdaptiveAvgPool1d(50)

        # decision layers
        # self.dc_conv1 = nn.Conv1d(self.out_planes, num_classes, kernel_size=1)
        # self.dc_bn1 = nn.BatchNorm1d(num_classes)
        # self.dc_se1 = nn.SELU()

        # self.dc_conv2 = nn.Conv1d(num_classes, 64, kernel_size=1)
        # self.dc_bn2 = nn.BatchNorm1d(64)
        # self.dc_se2 = nn.SELU()

        # self.dc_conv3 = nn.Conv1d(64, num_classes, kernel_size=1)
        # self.dc_bn3 = nn.BatchNorm1d(num_classes)

        # self.adaptiveAvgPool1d_2 = nn.AdaptiveAvgPool1d(1)

    def _forward_imp(self, x: torch.Tensor):

        out = self.conv1(x)

        mse1_out = self.mse1(out)
        mse2_out = self.mse2(mse1_out)
        mse3_out = self.mse3(mse2_out)

        uf1_out = self.uf1(mse3_out)
        # out = self.adaptiveAvgPool1d(out)

        # feature_out = self.dc_conv1(out)
        # out = self.dc_bn1(feature_out)
        # out = self.dc_se1(out)

        # embedded_out = self.dc_conv2(out)
        # out = self.dc_bn2(embedded_out)
        # out = self.dc_se2(out)

        # out = self.dc_conv3(out)
        # out = self.dc_bn3(out)

        # out = self.adaptiveAvgPool1d_2(out)
        # out = torch.flatten(out, 1)

        return uf1_out

    def forward(self, x: torch.Tensor):
        return self._forward_imp(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(size=(32, 10, 100))
    x = x.to(device=device)
    model = UFNet(in_planes=10, mse_expansions=[1, 1, 1], fft_flag=True, dwt_flag=True)
    model.to(device=device)
    res1 = model(x)
    print(res1.shape)