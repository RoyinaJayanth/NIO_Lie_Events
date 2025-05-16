import numpy as np
from metric import compute_ate_rte
import os

## this is how the data was saved

# np.save(osp.join(args.out_dir, data + '_gsn.npy'),
#                     np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))

## error was calculated 
#ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)

## now for ridi we apply procrustes on these trajectories and then compute ate and rte
def similarity_transform(from_points, to_points):
    
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    
    N, m = from_points.shape
    
    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)
    
    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m
    
    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
    
    cov_matrix = delta_to.T.dot(delta_from) / N
    
    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)
    
    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
    
    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c*R.dot(mean_from)
    
    return c*R, t

dir_path = '/home/royinakj/ronin/ev_data/ronin_ev_se3p/test_ridi_test_dist'

ate_all,rte_all = [],[]
pred_per_min = 200 * 60
for file in os.listdir(dir_path):
    if '_gsn.npy' in file:
        data = np.load(dir_path+'/'+file)
        pred = data[:,:2]
        targ = data[:,2:4]
        R,t = similarity_transform(pred,targ)
        pred = np.einsum('ij,tj->ti',R,pred) + t.reshape((1,-1))
        ate, rte = compute_ate_rte(pred[:,:2], targ[:,:2], pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
    else:
        continue
print(np.mean(ate_all), np.mean(rte_all), np.median(ate_all), np.median(rte_all))
