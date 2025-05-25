import torch
import numpy as np
from network.covariance_parametrization import DiagonalParam
import json

MIN_LOG_STD = np.log(1e-3)


"""
MSE loss between prediction and target, no logstdariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""


def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss


"""
Log Likelihood loss, with logstdariance (only support diag logstd)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_logstd meaning:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_logstd, targ):

    pred_logstd = torch.maximum(pred_logstd, MIN_LOG_STD * torch.ones_like(pred_logstd))
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
    return loss


"""
Log Likelihood loss, with logstdariance (support full logstd)
(NOTE: output is Nx1)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nxk logstdariance parametrization
output:
  loss: Nx1 vector of likelihood loss

resulting pred_logstd meaning:
DiagonalParam:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
PearsonParam:
pred_logstd (Nx6): u = [log(sigma_x) log(sigma_y) log(sigma_z)
                     rho_xy, rho_xz, rho_yz] (Pearson correlation coeff)
FunStuff
"""


def criterion_distribution(pred, pred_logstd, targ):
    loss = DiagonalParam.toMahalanobisDistance(
        targ, pred, pred_logstd, clamp_logstdariance=False
    )

def loss_full_inner_pdt_cov(pred, pred_logstd, targ, clamp_covariance = False):
    
    # compute the inverse of covariance matrices - just the lower triangular matrix of cholskey decomposition
    ## adding a small value to the diagonal elements to ensure matrix is non singular
    # pred_logstd = pred_logstd ## should not affect equivariance
    N = targ.shape[0]
    # u = torch.cholesky(pred_logstd)
    # CovInv = torch.cholesky_inverse(u)
    CovInv = torch.inverse(pred_logstd)

    

    # compute the error
    err = pred - targ
    loss_part1 = torch.einsum("ki,kij,kj->k", err, CovInv, err)
    if clamp_covariance:
        loss_part2 = torch.log(pred_logstd.det().clamp(min=1e-6))
    else:
        loss_part2 = torch.logdet(pred_logstd)
    
    

    loss = loss_part1 + loss_part2

    # if loss_part1.detach().mean()<0:
    #     print('covariance matrix shape:', pred_logstd.shape)
    #     print('cov det shape:', pred_logstd.detach().det().shape)
    #     print('det min:', pred_logstd.detach().det().min())
    #     print('det max:', pred_logstd.detach().det().max())
    #     print('det mean:', pred_logstd.detach().det().mean())
    #     print('loss_(logdet)_part2 mean:', loss_part2.detach().mean())
    #     print('loss mahalanobis distance mean:', loss_part1.detach().mean())

    #     print('loss mean:', loss.detach().mean())

        # store_d = {'covariance matrix shape':pred_logstd.shape,
        #            'cov det shape':}


    return loss.reshape((N, -1))

"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""
def get_loss(pred, pred_logstd, targ, epoch, arch_type):

    if epoch < 10 or '_only_mse' in arch_type: #10
        pred_logstd = pred_logstd.detach()
        loss = loss_mse(pred, targ)
    
    elif '_frameop' in arch_type or arch_type == 'resnet_fullCov_tlio_frame' or 'o2_frame_fullCov' in arch_type:
        loss = loss_distribution_diag(pred, pred_logstd, targ)
    elif '_6v_6s' in arch_type or '_fullCov' in arch_type or 'PearsonCov' in arch_type:
        loss = loss_full_inner_pdt_cov(pred, pred_logstd, targ, clamp_covariance=True)
    else:
        loss = loss_distribution_diag(pred, pred_logstd, targ)
    

    # if epoch < 10:s
    #     pred_logstd = pred_logstd.detach()s

    # loss = loss_distribution_diag(pred, pred_logstd, targ)
    return loss
