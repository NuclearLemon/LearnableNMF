import torch
from torch import nn
import torch.nn.functional as F


class LNMF(nn.Module):
    def __init__(self, m, n, b, r, l, init, rho, device):
        super(LNMF, self).__init__()
        self.m, self.n, self.r, self.layers, self.b = m, n, r, l, b
        self.device = device
        self.rho_w, self.rho_h = nn.ParameterList(), nn.ParameterList()
        self.w_para = nn.ParameterList()
        self.wtw_para = nn.ParameterList()
        init_v = float(rho)
        self.w_init = init
        wtw = self.w_init.T.mm(self.w_init)
        for k in range(self.layers):
            self.rho_w.append(nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device)))
            self.rho_h.append(nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device)))
            self.w_para.append(nn.Linear(self.m, self.r, bias=True, dtype=torch.float32, device=device))
            self.wtw_para.append(nn.Parameter(wtw.clone()))

    def forward(self, data):
        m, b = data.shape
        W = self.w_init.clone()
        H = torch.rand(self.r, b, dtype=torch.float32, device=self.device)
        X, Y = W.clone(), H.clone()
        Lamb_w = torch.zeros(m, self.r, dtype=torch.float32, device=self.device)
        Lamb_h = torch.zeros(self.r, b, dtype=torch.float32, device=self.device)
        param_dict = {
            'W': [],
            'H': [],
        }

        for k in range(self.layers):
            W = (data.mm(H.T) - Lamb_w + self.rho_w[k] * X).mm(
                torch.linalg.inv(H.mm(H.T) + self.rho_w[k] * torch.eye(self.r, device=self.device)))
            H = torch.linalg.inv(self.wtw_para[k] + self.rho_h[k] * torch.eye(self.r, device=self.device)).mm(
                self.w_para[k](data.T).T - Lamb_h + self.rho_h[k] * Y)
            X = F.relu(W + Lamb_w / self.rho_w[k])
            Y = F.relu(H + Lamb_h / self.rho_h[k])
            Lamb_w = Lamb_w + self.rho_w[k] * (W - X)
            Lamb_h = Lamb_h + self.rho_h[k] * (H - Y)
            param_dict['W'].append(W)
            param_dict['H'].append(H)

        return param_dict['W'], param_dict['H']

    def name(self):
        return "Learnable NMF"

    def get_obj(self, V, W, H):
        return (torch.norm(V - W.mm(H), p='fro') ** 2) / (2 * self.n)
