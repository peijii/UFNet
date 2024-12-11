import torch
import torch.nn as nn
from torch.autograd import Function
import pywt


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
        coeffs = []
        for _ in range(self.levels):
            x = self.dwt(x)
            coeffs.append(x[:, x.shape[1] // 2:, :])  # Save high-frequency components
            x = x[:, :x.shape[1] // 2, :]  # Low-frequency components for next level
        coeffs.append(x)  # Final low-frequency component
        return coeffs

class IDWTLayer(nn.Module):
    def __init__(self, wave, levels):
        super(IDWTLayer, self).__init__()
        self.levels = levels
        self.idwt = IDWT1D(wave)

    def forward(self, coeffs):
        x = coeffs[-1]  # Start with the final low-frequency component
        for i in range(self.levels - 1, -1, -1):
            x = torch.cat([x, coeffs[i]], dim=1)  # Combine low and high-frequency components
            x = self.idwt(x)  # Perform inverse DWT
        return x
    

class LDWT(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass
    



# Example usage
if __name__ == "__main__":
    wave = 'haar'  # Daubechies 1 wavelet
    dwt = DWTLayer(wave='haar', levels=2)
    idwt = IDWTLayer(wave='haar', levels=2)

    # Input time series (batch_size=8, channels=1, length=64)
    x = torch.randn(10, 10, 200)

    # Perform DWT and IDWT
    coeffs = dwt(x)
    reconstructed = idwt(coeffs)

    print("Original shape:", x.shape)
    #print("DWT coefficients shape:", coeffs.shape)
    print("Reconstructed shape:", reconstructed.shape)
