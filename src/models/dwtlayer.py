import imp
from typing import List, Tuple
import torch.nn as nn
import pywt
import torch
import decomposition as depo


class DWT1DForward(nn.Module):

    def __init__(
        self,
        levels: int = 1,
        wave: str = 'db1',
        mode: str = 'zero'
    ) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wave = pywt.Wavelet(wave)
        low_pass, high_pass = wave.dec_lo, wave.dec_hi
        filters = depo.prep_filt_afb1d(h0=low_pass, h1=high_pass)
        self.h0 = filters[0].to(device)
        self.h1 = filters[1].to(device)
        self.levels = levels
        self.mode = mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, L_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients.
        """
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        mode = depo.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.levels):
            x0, x1 = depo.AFB1D.apply(x0, self.h0, self.h1, mode)
            highs.append(x1)

        return x0, highs


class DWT1DInverse(nn.Module):
    """ Performs a 1d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(
        self,
        wave: str = 'db1',
        mode: str = 'zero'
    ) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wave = pywt.Wavelet(wave)
        low_pass, high_pass = wave.dec_lo, wave.dec_hi
        # Prepare the filters
        filters = depo.prep_filt_sfb1d(low_pass, high_pass)
        self.g0 = filters[0].to(device)
        self.g1 = filters[1].to(device)
        self.mode = mode

    def forward(self, coeffs: Tuple[torch.Tensor, List]) -> torch.Tensor:
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, should
              match the format returned by DWT1DForward.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, L_{in})`

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = depo.mode_to_int(self.mode)
        # Do a multilevel inverse transform
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = depo.SFB1D.apply(x0, x1, self.g0, self.g1, mode)
        return x0


class DWTLayer(nn.Module):

    def __init__(
        self,
        levels: int = 1
    ) -> None:
        super(DWTLayer, self).__init__()
        self.dwt = DWT1DForward(levels=levels)
        self.idwt = DWT1DInverse()
        self.low_pass_parameter = nn.Parameter(torch.ones(size=(1, ), dtype=torch.float32))
        self.high_pass_parameter = nn.Parameter(torch.ones(size=(levels+1, ), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_x, high_xs = self.dwt(x)
        low_x = self.low_pass_parameter * low_x
        high_xs = [high_x * self.high_pass_parameter[i] for i, high_x in enumerate(high_xs)]
        x = self.idwt((low_x, high_xs))
        return x


if __name__ == '__main__':
    x = torch.randn(size=(10, 10, 200))
    model = DWTLayer(levels=1)
    res = model(x)
    print(res.shape)