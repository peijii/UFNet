import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class FFTLayer(nn.Module):

    def __init__(
        self,
        in_planes : int,
        length: int,
        wfb_switch: bool = False,
        N: int = 3
    ):
        super(FFTLayer, self).__init__()
        self.wfb_switch = wfb_switch
        if not self.wfb_switch:
            self.parameter = nn.Parameter(torch.randn(size=(in_planes, int(length // 2) + 1, 2), dtype=torch.float32) * 1.0, requires_grad=True)
        else:
            self.parameter = nn.Parameter(torch.randn(size=(N, in_planes, int(length // 2) + 1, 2), dtype=torch.float32) * 1.0, requires_grad=True)
            self.reweight_R = CnnEncoder(in_channels=in_planes, out_channels=N, ratio=2)
            self.reweight_I = CnnEncoder(in_channels=in_planes, out_channels=N, ratio=2)

    def forward_fft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        fft_x = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.parameter)
        fft_x = fft_x * weight
        x_star = torch.fft.irfft(fft_x, dim=-1, n=L, norm='ortho')
        return x_star
    
    def forward_wfb_fft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        fft_x = torch.fft.rfft(x, dim=-1, norm='ortho')
        #weight = torch.view_as_complex(self.parameter)
        W_R, W_I = self.parameter[:, :, :, 0], self.parameter[:, :, :, 1]
        X_R, X_I = fft_x.real, fft_x.imag
        C_R = self.reweight_R(X_R)
        C_I = self.reweight_I(X_I)
        Kf = torch.complex((C_R[..., None, None] * W_R).sum(dim=1), (C_I[..., None, None] * W_I).sum(dim=1))
        fft_x = fft_x * Kf
        x_star = torch.fft.irfft(fft_x, dim=-1, n=L, norm='ortho')
        return x_star
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.wfb_switch:
            return self.forward_fft(x)
        return self.forward_wfb_fft(x)


class CnnEncoder(nn.Module):

    def __init__(self, in_channels, out_channels=3, ratio=4, kernel_size=3, drop_prob=0.2):
        super().__init__()
        hidden_channels = int(ratio * in_channels)

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.SELU(inplace=True)  
        self.drop1 = nn.Dropout(drop_prob) 
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)  
        x = self.act(x)  
        x = self.drop1(x) 
        x = self.conv2(x) 
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    x = torch.randn(size=(10, 10, 200))
    plt.plot(x[0][0])
    model = FFTLayer(in_planes=10, length=200, wfb_switch=True)
    res = model(x)
    plt.plot(res[0][0].detach().numpy())
    plt.show()
    print(res[0][0].detach().numpy())