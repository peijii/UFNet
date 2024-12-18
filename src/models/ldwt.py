import torch
import torch.nn as nn
from torch.autograd import Function
import pywt
from typing import Union, TypeVar, Tuple
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


class MLWaveLetProcessor(nn.Module):

    def __init__(self, in_channels, length, levels=2, ratio=2, kernel_size=3):
        super(MLWaveLetProcessor, self).__init__()
        # Compute lengths for each level
        self.lengths = self.compute_lengths(length, levels)
        self.mlwaveletprocessor = nn.ModuleDict({
            f"conv_{length}": nn.Sequential(
                nn.Conv1d(in_channels, in_channels*ratio, kernel_size, padding=kernel_size // 2, groups=1),
                nn.SELU(inplace=True),
                nn.Conv1d(in_channels*ratio, in_channels, kernel_size, padding=kernel_size // 2, groups=1),
                nn.Sigmoid()
            ) for length in self.lengths})

    def compute_lengths(self, length, levels):
        """
        Compute the lengths for each level.
        """
        lengths = []
        for _ in range(levels):
            length //= 2
            lengths.append(length)
        lengths.append(length)
        return lengths

    def forward(self, x):
        outputs = []
        for idx in range(len(self.lengths)):
            conv = self.mlwaveletprocessor[f"conv_{self.lengths[idx]}"]
            output = conv(x[idx])
            outputs.append(output)
        return outputs
    

class DWT_1D_Function(Function):
    @staticmethod
    def forward(ctx, x, w_low, w_high):
        x = x.contiguous()

        w_low = w_low.to(dtype=x.dtype)
        w_high = w_high.to(dtype=x.dtype)

        ctx.save_for_backward(w_low, w_high)
        ctx.shape = x.shape

        dim = x.shape[1]
        # Apply low-pass and high-pass filters
        x_low = torch.nn.functional.conv1d(x, w_low.expand(dim, -1, -1), stride=2, groups=dim)
        x_high = torch.nn.functional.conv1d(x, w_high.expand(dim, -1, -1), stride=2, groups=dim)
        # Concatenate low and high frequency components
        x = torch.cat([x_low, x_high], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_low, w_high = ctx.saved_tensors
            B, C, L = ctx.shape
            dx = dx.view(B, 2, -1, L // 2).transpose(1, 2).reshape(B, -1, L // 2)

            filters = torch.cat([w_low, w_high], dim=0).repeat(C, 1, 1)
            dx = torch.nn.functional.conv_transpose1d(dx, filters, stride=2, groups=C)
        return dx, None, None


class IDWT_1D_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        filters = filters.to(dtype=x.dtype)
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, L = x.shape
        x = x.view(B, 2, -1, L).transpose(1, 2).reshape(B, -1, L)
        filters = filters.repeat(x.shape[1] // 2, 1, 1)

        x = torch.nn.functional.conv_transpose1d(x, filters, stride=2, groups=x.shape[1] // 2)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            filters = filters.to(dtype=dx.dtype)

            B, C, L = ctx.shape
            dx = dx.contiguous()

            w_low, w_high = torch.unbind(filters, dim=0)
            x_low = torch.nn.functional.conv1d(dx, w_low.unsqueeze(1).expand(C // 2, -1, -1), stride=2, groups=C // 2)
            x_high = torch.nn.functional.conv1d(dx, w_high.unsqueeze(1).expand(C // 2, -1, -1), stride=2, groups=C // 2)

            dx = torch.cat([x_low, x_high], dim=1)
        return dx, None


class DWT1D(nn.Module):
    def __init__(self, wave):
        super(DWT1D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_low = torch.Tensor(w.dec_lo[::-1])
        dec_high = torch.Tensor(w.dec_hi[::-1])

        w_low = dec_low.unsqueeze(0).unsqueeze(1)
        w_high = dec_high.unsqueeze(0).unsqueeze(1)

        self.register_buffer('w_low', w_low)
        self.register_buffer('w_high', w_high)

    def forward(self, x):
        return DWT_1D_Function.apply(x, self.w_low, self.w_high)


class IDWT1D(nn.Module):
    def __init__(self, wave):
        super(IDWT1D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_low = torch.Tensor(w.rec_lo)
        rec_high = torch.Tensor(w.rec_hi)

        w_low = rec_low.unsqueeze(0).unsqueeze(1)
        w_high = rec_high.unsqueeze(0).unsqueeze(1)

        filters = torch.cat([w_low, w_high], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        return IDWT_1D_Function.apply(x, self.filters)
        

class DWTLayer(nn.Module):
    def __init__(self, wave, levels):
        super(DWTLayer, self).__init__()
        self.levels = levels
        self.dwt = DWT1D(wave)

    def forward(self, x):
        coeffs_list = []
        for _ in range(self.levels):
            x = self.dwt(x)
            # Split low and high frequencies
            C = x.shape[1] // 2
            x_low, x_high = x[:, :C, :], x[:, C:, :]
            coeffs_list.append(x_high)  # Store high frequencies
            x = x_low  # Use low frequency for next level
        coeffs_list.append(x)  # Add final low frequency component
        return coeffs_list


class IDWTLayer(nn.Module):
    def __init__(self, wave, levels):
        super(IDWTLayer, self).__init__()
        self.levels = levels
        self.idwt = IDWT1D(wave)

    def forward(self, coeffs_list):
        if self.levels == 1:
            x_low = coeffs_list[-1] 
            x_high = coeffs_list[0]
            x = torch.cat([x_low, x_high], dim=1)
            return self.idwt(x)
        else:
            x = coeffs_list[-1] 
            for i in range(self.levels - 1, -1, -1):
                x_high = coeffs_list[i]
                if x_high.shape[-1] != x.shape[-1]:
                    x = torch.nn.functional.pad(x, (0, 1))
                x = torch.cat([x, x_high], dim=1) 
                x = self.idwt(x)
            return x


class LDWT(nn.Module):

    def __init__(
            self,
            in_planes : int,
            length: int,
            level: int = 1,
            att_switch: bool = False
    ):
        super().__init__()
        self.parameter = []
        self.length = length
        self.level = level
        self.att_switch = att_switch
        if self.att_switch:
            self.mwp = MLWaveLetProcessor(in_channels=in_planes, length=self.length, levels=self.level, ratio=2, kernel_size=3)
            for _ in range(level):
                length //= 2
                self.parameter.append(nn.Parameter(torch.randn(size=(in_planes, length), dtype=torch.float32) * 1.0, requires_grad=True))
            self.parameter.append(nn.Parameter(torch.randn(size=(in_planes, length), dtype=torch.float32) * 1.0, requires_grad=True))
        else:
            for _ in range(level):
                length //= 2
                self.parameter.append(nn.Parameter(torch.randn(size=(in_planes, length), dtype=torch.float32) * 1.0, requires_grad=True))
            self.parameter.append(nn.Parameter(torch.randn(size=(in_planes, length), dtype=torch.float32) * 1.0, requires_grad=True))

        self.dwt = DWTLayer(wave='haar', levels=level)
        self.idwt = IDWTLayer(wave='haar', levels=level)

    def forward_dwt(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)
        x = [x[i] * self.parameter[i] for i in range(self.level+1)]
        x_star = self.idwt(x)
        return x_star

    def forward_att_dwt(self, x):
        x_dwt = self.dwt(x)
        x = self.mwp(x_dwt)
        x = [x[i] * self.parameter[i] for i in range(self.level+1)]
        x = [x[i] * x_dwt[i] for i in range(self.level+1)]
        x_star = self.idwt(x)
        return x_star
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.att_switch:
            return self.forward_att_dwt(x)
        else:
            return self.forward_dwt(x)
    

# Example usage
if __name__ == "__main__":
    x = torch.randn(10, 10, 100)
    model = LDWT(in_planes=10, length=100, level=5, att_switch=False)
    res = model(x)
    print(res.shape)