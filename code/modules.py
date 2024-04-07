import torch.nn as nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class ComplexConv(nn.Module):
    def __init__(self, inchnl, outchnl, kernel, dilation, groups):
        super(ComplexConv, self).__init__()
        self.conv = nn.Conv1d(inchnl, outchnl, kernel, padding='same', dilation=dilation, groups=groups, bias=False, dtype=torch.cfloat)

    def forward(self, x):
        x0 = x.unsqueeze(0).permute(0, 2, 1)
        y = self.conv(x0).permute(0, 2, 1).squeeze(0)
        return y

class LUT1D(nn.Module):
    def __init__(self, nspline, nbranch):
        super(LUT1D, self).__init__()
        self.nspline = nspline
        self.nbranch = nbranch
        self.weight = nn.Parameter(torch.zeros(nbranch, nspline, 1, dtype=torch.cfloat))

    def forward(self, x):
        x_abs = x.abs().T.unsqueeze(-1) * (self.nspline - 1)
        splines = torch.relu(1 - (x_abs - torch.arange(self.nspline).to(device)).abs())
        z = splines.cfloat().matmul(self.weight).squeeze(-1).T
        return z * x