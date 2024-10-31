import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FFTLayer(nn.Module):

    def __init__(
        self,
        in_planes : int,
        length: int
    ):
        super(FFTLayer, self).__init__()
        self.parameter = nn.Parameter(torch.randn(size=(in_planes, length, 2), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_x = torch.fft.fft(x, dim=2, norm='ortho')
        weight = torch.view_as_complex(self.parameter)
        fft_x = fft_x * weight
        x = torch.fft.ifft(fft_x, dim=2, norm='ortho')
        return x.real

if __name__ == '__main__':
    x = torch.randn(size=(10, 10, 200))
    plt.plot(x[0][0])
    model = FFTLayer(dim=10, length=200)
    res = model(x)
    plt.plot(res[0][0].detach().numpy())
    plt.show()
    print(res[0][0].detach().numpy())