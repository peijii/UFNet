import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class FFTModule(nn.Module):

    def __init__(self, dim, length):
        super(FFTModule, self).__init__()
        self.parameter = nn.Parameter(torch.randn(size=(dim, length, 2), dtype=torch.float32))

    def forward(self, x):
        fft_x = torch.fft.fft(x, dim=2, norm='ortho')
        weight = torch.view_as_complex(self.parameter)
        fft_x = fft_x * weight
        x = torch.fft.ifft(fft_x, dim=2, norm='ortho')
        return x



if __name__ == '__main__':
    x = torch.randn(size=(10, 10, 200))
    raw_x11 = x[0][0]
    plt.plot(raw_x11)
    model = FFTModule(dim=10, length=200)
    res = model(x).detach().numpy()
    #print(res[0][0])
    plt.plot(res[0][0])
    plt.show()