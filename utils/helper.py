from calendar import c
from turtle import forward
from cv2 import normalize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MSE loss for NeRF rendering
def img2mse(x, y):
    return torch.mean((x - y) ** 2)

# L2 loss function for under & over exposure conditions
def img2mse_gamma(x, y, gamma=2, type='under'):
    eta = 1e-4
    if type == 'under':
        return torch.mean(((x+eta)**(1/gamma) - (y+eta)**(1/gamma)) ** 2)
    elif type == 'over':
        return torch.mean((x**gamma - y**gamma) ** 2)


# inverse tone curve MSE loss
def img2mse_tone(x, y):
    eta=1e-4
    x = torch.clip(x, min = eta, max = 1-eta)
    # the inverse tone curve, pls refer to paper (Eq.13): 
    # "https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf"
    f=lambda x: 0.5 - torch.sin(torch.asin(1.0 - 2.0 * x) / 3.0)
    # f=lambda x: 1 - torch.sin(torch.asin(1.0 - 3.0 * x) / 3.0)
    fy = f(y)
    # fy_gamma, gamma = auto_gamma_new(fy)
    fx = f(x)
    # return torch.mean(((fx**gamma) - fy_gamma) ** 2)
    return torch.nn.functional.l1_loss((fx), fy)


# Gray World Colour Constancy
def colour(x):
    Drg = torch.pow(x[0]-x[1], 2)
    Drb = torch.pow(x[1]-x[2], 2)
    Dgb = torch.pow(x[2]-x[0], 2)
    k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
    return k

# TV loss 
def Smooth_loss(x):
    x = x.squeeze(-1)
    x_left = torch.mean(x[:, :-2], dim=0)
    x_middle = torch.mean(x[:, 1:-1], dim=0)
    x_right = torch.mean(x[:, 2:], dim=0)
    g_pp = -2*x_middle + x_left + x_right
    smooth_loss = torch.sum(torch.square(g_pp))
    return smooth_loss

def L1_loss(x, y):
    return torch.abs(x - y)

class Structure_Loss(nn.Module):
    def __init__(self, contrast):
        super(Structure_Loss, self).__init__()
        self.kernel_left = torch.FloatTensor([-1,1,0]).unsqueeze(0).unsqueeze(0)
        self.kernel_right = torch.FloatTensor([0,1,-1]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=self.kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=self.kernel_right, requires_grad=False)
        self.pool = nn.AvgPool1d(4)
        self.contrast = contrast
    
    def forward(self, low, high):
        # dim 0 is 3 
        low, high = low.permute(1,0).unsqueeze(0), high.permute(1,0).unsqueeze(0)
        
        # dim 1 is 3
        low_mean, high_mean = torch.mean(low, dim=1, keepdim=True), torch.mean(high, dim=1, keepdim=True)
        
        low_pool, high_pool = self.pool(low_mean), self.pool(high_mean)

        low_left = F.conv1d(low_pool, self.weight_left.to(low_pool.device), padding=1)
        low_right = F.conv1d(low_pool, self.weight_right.to(low_pool.device), padding=1)

        high_left = F.conv1d(high_pool, self.weight_left.to(high_pool.device), padding=1)
        high_right = F.conv1d(high_pool, self.weight_right.to(high_pool.device), padding=1)

        D_left, D_right = torch.pow(self.contrast*low_left - high_left,2), torch.pow(self.contrast*low_right - high_right, 2)
        
        return torch.mean(D_left + D_right)
        

class Exp_loss(nn.Module):
    def __init__(self, patch_size=64, mean_val=0.2):
        super(Exp_loss, self).__init__()
        self.pool = nn.AvgPool1d(patch_size)
        self.mean_val = mean_val
    
    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True).permute(2,1,0)
        mean = self.pool(x)
        loss = torch.mean(torch.pow((mean-self.mean_val), 2))
        return loss

class Exp_loss_global(nn.Module):
    def __init__(self, mean_val=0.5):
        super(Exp_loss_global, self).__init__()
        self.mean_val = mean_val
    
    def forward(self, x):
        #x = torch.mean(x, 1, keepdim=True).permute(2,1,0)
        # (N, 1) -> N, 1, 1 -> 1, 1, N
        x = torch.mean(x,-1,keepdim=True).unsqueeze(-1).permute(2,1,0)
        mean = self.global_average_pool(x)
        loss = torch.pow(torch.mean((mean-self.mean_val)), 2)
        return loss

    # global average pooling
    def global_average_pool(self, x):
        return F.avg_pool1d(x, x.size(2))



def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :] 




def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def sample_pdf(bins, weights, origins, directions, t_vals, num_samples, randomized):

    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values
    coords = cast_rays(t_vals, origins, directions)
    return t_vals, coords


if __name__ == '__main__':
    low = torch.randn([2048, 3])
    high = torch.randn([2048, 3])
    str_loss = Structure_Loss()
    loss = str_loss(low, high)



