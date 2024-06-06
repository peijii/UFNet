from turtle import forward
from typing import Tuple
import torch.nn as nn
import pywt
import torch
import numpy as np
import decomposition as depo

class DWT1DForward(nn.Module):

    def __init__(
        self,
        levels: int = 1,
        wave: str = 'db1',
        mode: str = 'zero'
    ) -> None:
        super().__init__()
        wave = pywt.Wavelet(wave)
        low_pass, high_pass = wave.dec_lo, wave.dec_hi
        filters = depo.prep_filt_afb1d(h0=low_pass, h1=high_pass)
        self.h0 = filters[0]
        self.h1 = filters[1]
        self.levels = levels
        self.mode = mode

    def forward(self, x):
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
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]

        # Prepare the filters
        filts = depo.prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])
        self.register_buffer('g1', filts[1])
        self.mode = mode

    def forward(self, coeffs):
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


if __name__ == '__main__':
    x = torch.randn(size=(10, 10, 1000))
    raw_x11 = x[0][0]
    model = DWT1DForward(levels=3)
    res = model(x)
    print(res[0].shape)
    #print(res[0][0])