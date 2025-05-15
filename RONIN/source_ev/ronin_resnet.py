import os
import time
from os import path as osp

import numpy as np
import torch
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_glob_speed import *
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *
from data_glob_speed_test import *
from tqdm import tqdm

import time
import math
from scipy.spatial.transform import Rotation

import numpy as np
from utils_tlio.math_utils import mat_exp, mat_log, exp_SE3 as mat_exp_se3, log_SE3 as mat_log_se3 # type: ignore

from numba import jit
from scipy.special import erfinv

_input_channel, _output_channel = 12, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
_contrast_threshold = 0.1
_initial_velocity = np.zeros((1,3))



def hat(v):
    v = np.squeeze(v)
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R


@jit(nopython=True)
def vee(w_x):
    return np.array([w_x[2, 1], w_x[0, 2], w_x[1, 0]])


@jit(nopython=True)
def mat_exp(omega):
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    angle = np.linalg.norm(omega)

    # Near phi==0, use first order Taylor expansion
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)


@jit(nopython=True)
def sinc_robust(x):
    if np.abs(x) < 1e-3:
        return 1
    else:
        return np.sin(x) / x


@jit(nopython=True)
def mat_log(R):
    x = 0.5 * (np.trace(R) - 1) 
    if x > 1:
        x = 1
    elif x < -1:
        x = -1
    theta = np.arccos(x)  # - 1e-14
    if np.abs(theta - np.pi) < 1e-3:
        return np.zeros((3,))
    omega = vee(R - R.T) * 0.5 / sinc_robust(theta)
    return omega


@jit(nopython=True)
def exp_SE3(v):
    """
    Aligns with the Sophus convention of the 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    Exp = np.eye(4)
    Exp[:3, :3] = mat_exp(v[3:])
    Exp[:3, 3:4] = Jl_SO3(v[3:]) @ v[:3, None]
    return Exp


#@jit(nopython=True)
def log_SE3(T):
    """
    Aligns with the Sophus convention of the returned 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    w = mat_log(T[:3, :3])
    V_inv = Jl_SO3_inv(w)
    v = V_inv @ T[:3, 3:4]
    out_v = np.zeros((6,))
    out_v[0:3] = v[:, 0]
    out_v[3:] = w
    return out_v

def Jr_log(phi):
    """ right jacobian for log operation on SO(3) """
    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        J = np.eye(3) + 0.5 * hat(phi)
    else:
        J = (
                np.eye(3)
                + 0.5 * hat(phi)
                + (
                        1 / np.power(theta, 2.0)
                        + (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
                )
                * hat(phi)
                * hat(phi)
        )
    return J


def Jr_SO3(phi):
    """ right Jacobian of SO(3) (Eq. 7.77a in Barfoot's "State Estimation in Robotics" book) """
    phi_norm = np.linalg.norm(phi)
    if phi_norm < 1e-5:
        return np.eye(3)
    else:
        a = phi / phi_norm
        a = a.reshape(3, 1)
        sin_phi_div_phi = np.sin(phi_norm) / phi_norm
        return sin_phi_div_phi * np.eye(3) + (1 - sin_phi_div_phi) * a @ a.T + (1 - np.cos(phi_norm)) / phi_norm * hat(
            a)


@jit(nopython=True)
def Jl_SO3(phi):
    """ Left Jacobian of SO(3) """

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    theta = np.linalg.norm(phi)
    Om = np.ascontiguousarray(hat(phi))
    if theta < 1e-5:
        return np.eye(3) + 0.5 * Om
    else:
        theta2 = theta ** 2
        return np.eye(3) + (1 - np.cos(theta)) / theta2 * Om + (theta - np.sin(theta)) / (theta2 * theta) * Om @ Om

@jit(nopython=True)
def Jl_SO3_inv(phi):  ## for numba
    """ Inverse of left Jacobian of SO(3) """

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return R

    theta = np.sqrt(phi[0] ** 2 + phi[1] ** 2 + phi[2] ** 2)
    Om = hat(phi)

    # Manually define the identity matrix
    # identity = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 1.0]
    # ])

    if theta < 1e-5:
        return np.eye(3, dtype=np.float64) - float(0.5) * Om + float(1.0 / 12) * np.dot(Om, Om)
    else:
        theta2 = theta ** 2
        return np.eye(3, dtype=np.float64) - 0.5 * Om + (1 - 0.5 * theta * np.cos(theta / 2) / np.sin(theta / 2)) / theta ** 2 * np.dot(Om, Om)

@jit(nopython=True)
def mat_exp_se3(v):
    """
    Aligns with the Sophus convention of the 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    Exp = np.eye(4)
    Exp[:3,:3] = mat_exp(v[3:])
    Exp[:3,3:4] = Jl_SO3(v[3:]) @ np.ascontiguousarray(v[:3,None])
    return Exp

@jit(nopython=True)
def _geodesic_events_se3_vectorized(T_ref, T0, T1, t0, t1, threshold, gyro_01=None, accel_01=None, add_polarity=False):
    # Imagine starting at T_ref, observing T0, T1.
    # Take steps of size threshold toward T1 until this is no longer possible
    # Each time, the time is measured according to the projection along the shortest path between T1 and T0.
    # We need to do this because T0 starts at t0, and T1 is at t1
    # print(T_ref.data.contiguous, T0.data.contiguous, T1.data.contiguous)


    dT_0_1_alg = mat_log_se3(np.linalg.inv(T0) @ T1)
    n_w01 = (dT_0_1_alg / np.linalg.norm(dT_0_1_alg)).astype("float64")

    T_ref_1 = np.linalg.inv(T_ref) @ T1
    w_ref_1 = mat_log_se3(T_ref_1)

    polarity = w_ref_1 / np.linalg.norm(w_ref_1)

    step = threshold * polarity
    n_events = int(np.floor(np.linalg.norm(w_ref_1 / threshold)))

    num_dim = 7
    if add_polarity:
        num_dim += 6

    events = np.zeros((n_events, num_dim))

    steps = (np.arange(n_events, dtype=np.float64) + 1)[:,None] * step[None, :]

    mat_exp_steps = mat_exp_se3_vectorized(steps).astype("float64")

    T0_inv = np.linalg.inv(T0)
    dT_refs = np.empty((len(mat_exp_steps), 4, 4), dtype=np.float64)
    T_refs = np.empty((len(mat_exp_steps), 4, 4), dtype=np.float64)
    polarities = np.empty((len(mat_exp_steps), 6), dtype=np.float64)
    ## first polarity is wrt first T_ref
    pol_T_ref = T_ref.copy()
    for i in range(len(T_refs)):
        T_refs[i] = T_ref @ mat_exp_steps[i]
        dT_refs[i] = T0_inv @ T_refs[i]
        R_ref = np.ascontiguousarray(pol_T_ref[:3,:3])
        polarities[i,:3] = (R_ref @ polarity[:3])[None,]
        polarities[i,3:] = (R_ref @ polarity[3:])[None,]
        pol_T_ref = T_refs[i].copy()

    dT_0_ref_algs = mat_log_se3_vectorized(dT_refs)

    bils = dT_0_ref_algs.dot(np.ascontiguousarray(n_w01[:,None]))[:,0] / np.linalg.norm(dT_0_1_alg)
    t_i_s = t0 + bils * (t1 - t0)

    events[:,0] = t_i_s


    if gyro_01 is not None:
        events[:, 1:4] = gyro_01[0][None,:] + (gyro_01[1] - gyro_01[0])[None,:] * bils[:,None]
        events[:, 4:7] = accel_01[0][None,:] + (accel_01[1] - accel_01[0])[None,:] * bils[:,None]
    else:
        events[:, 1:4] = mat_log_vectorized(T_refs[:,:3, :3])
        events[:, 4:7] = T_refs[:,:3, 3]

    if add_polarity:
        events[:,-6:] = polarities #step[None, :]

    T_ref = T_ref @ mat_exp_se3(n_events * step).astype("float64")

    return T_ref, events

@jit(nopython=True)
def Jl_SO3_inv_vectorized(phi):
    """ Inverse of left Jacobian of SO(3) """

    theta = np.sqrt(phi[:,0:1]**2+phi[:,1:2]**2+phi[:,2:3]**2)
    Om = hat_vectorized(phi)

    # Manually define the identity matrix
    # identity = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 1.0]
    # ])
    mask = theta[:,0] < 1e-5

    Om2 = np.empty_like(Om)
    for i,o in enumerate(Om):
        o = np.ascontiguousarray(o)
        Om2[i] = o @ o


    output = np.zeros((len(phi), 3, 3))
    output[mask] = np.eye(3, dtype=np.float64)[None,:,:] - 0.5 * Om[mask] + 1.0 / 12 * Om2[mask]
    theta = theta[..., None]
    output[~mask] =  (np.eye(3, dtype=np.float64)[None,:,:] - 0.5 * Om + (1 - 0.5 * theta * np.cos(theta / 2) / np.sin(theta / 2)) / theta ** 2 * Om2)[~mask]
    return output


@jit(nopython=True)
def Jl_SO3_vectorized(phi):
    """ Left Jacobian of SO(3) """

    theta = np.sqrt(phi[:,0:1]**2+phi[:,1:2]**2+phi[:,2:3]**2)
    Om = hat_vectorized(phi)

    output = np.zeros((len(phi), 3, 3))
    output[theta[:,0] < 1e-5]  = np.eye(3)[None,:,:] + 0.5 * Om[theta[:,0] < 1e-5]

    theta = theta[..., None]
    theta2 = theta ** 2

    term1 = np.eye(3)[None,:,:]
    term2 = (1 - np.cos(theta)) / theta2 * Om
    term3 = ((theta - np.sin(theta)) / (theta2 * theta))
    out = np.empty((len(phi), 3, 3))
    for i, o in enumerate(Om):
        o = np.ascontiguousarray(o)
        out[i,:,:] = o @ o
    term3 = term3 * out
    res = term1 + term2 + term3
    output[theta[:,0,0] >= 1e-5]  = res[theta[:,0,0] >= 1e-5]

    return output

@jit(nopython=True)
def sinc_robust_vectorized(x):
    output = np.zeros_like(x)
    mask = np.abs(x) < 1e-3
    output[mask] = 1
    output[~mask] = (np.sin(x)/x)[~mask]
    return output

@jit(nopython=True)
def mat_log_se3_vectorized(T):
    """
    Aligns with the Sophus convention of the returned 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    w = mat_log_vectorized(T[:,:3,:3]).astype(np.float64)
    V_inv = Jl_SO3_inv_vectorized(np.ascontiguousarray(w)).astype("float64")
    result = np.empty((len(T), 6), dtype=np.float64)
    R = np.ascontiguousarray(T[:,:3,3])
    for i in range(len(T)):
        result[i,:3] = V_inv[i] @ R[i]
        result[i,3:] = w[i]
    return result

@jit(nopython=True)
def hat_vectorized(vectors):
    n = vectors.shape[0]
    skew = np.zeros((n, 3, 3))

    # Fill in the upper and lower triangular parts of the skew-symmetric matrices
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]

    return skew


@jit(nopython=True)
def mat_exp_vectorized(omega):
    if omega.shape[-1] != 3:
        raise ValueError("tangent vector must have length 3")

    angle = np.sqrt(omega[:,0:1]**2+omega[:,1:2]**2+omega[:,2:3]**2)

    # Near phi==0, use first order Taylor expansion
    output = np.zeros((len(omega), 3, 3))
    output[angle[:,0] < 1e-10] = np.identity(3)[None, :, :] + hat_vectorized(omega[angle[:,0] < 1e-10])

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    res = c[...,None] * np.identity(3)[None, :, :] + (1 - c)[...,None] * axis[:,:,None]* axis[:,None,:] + s[...,None] * hat_vectorized(axis)
    output[angle[:,0] >= 1e-10] = res[angle[:,0] >= 1e-10]

    return output


@jit(nopython=True)
def vee_vectorized(w_x):
    output = np.zeros((len(w_x), 3))
    output[:,0] = w_x[:, 2, 1]
    output[:,1] = w_x[:, 0, 2]
    output[:,2] = w_x[:, 1, 0]
    return output


@jit(nopython=True)
def mat_log_vectorized(R):
    x = 0.5 * (R[:,0,0]+R[:,1,1]+R[:,2,2] - 1)  # np.clip(0.5*(np.trace(R) - 1), -1, 1)
    x[x>1] = 1
    x[x<-1] = -1

    theta = np.arccos(x)  # - 1e-14

    output = np.zeros((len(R), 3))
    output[np.abs(theta - np.pi) < 1e-3] = 0
    res = vee_vectorized(R - np.transpose(R,axes=(0,2,1))) * 0.5 / sinc_robust_vectorized(theta)[:,None]
    output[np.abs(theta - np.pi) >= 1e-3] = res[np.abs(theta - np.pi) >= 1e-3]

    return output

@jit(nopython=True)
def mat_exp_se3_vectorized(v):
    # n x 6
    Exp = np.zeros((len(v), 4, 4))
    Exp[:,-1,-1] = 1
    Exp[:, :3, :3] = mat_exp_vectorized(v[:, 3:])

    jl_vec = Jl_SO3_vectorized(np.ascontiguousarray(v[:, 3:]))
    for i in range(len(jl_vec)):
        Exp[i,:3,3] = jl_vec[i] @ v[i,:3]
    return Exp

@jit(nopython=True)
def mat_log_se3(T):
    """
    Aligns with the Sophus convention of the returned 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    w = mat_log(T[:3,:3])
    V_inv = Jl_SO3_inv(w).astype("float64")
    v = np.dot(V_inv, T[:3,3:4].astype("float64"))
    out_v = np.zeros((6,))
    out_v[0:3] = v[:,0]
    out_v[3:] = w
    # return np.concatenate([v[:,0], w], 0)
    return out_v

def geodesic_distance(R0, R1, t0, t1, b):
    dt = t1 - t0
    dr = mat_log(R0 @ R1.T)
    d = np.sqrt((dt**2).sum() + b * (dr**2).sum())
    return d

@jit(nopython=True)
def _preintegrate_measurement(R, p, v, gyr, a, dt):
    g = np.array([0, 0, -9.81])
    dtheta = gyr * dt
    dRd = mat_exp(dtheta)
    Rd = R @ dRd
    dv_w = R @ a * dt
    dp_w = 0.5 * dv_w * dt
    gdt = g * dt
    gdt22 = 0.5 * gdt * dt
    vd = v + dv_w + gdt
    pd = p + v * dt + dp_w + gdt22
    return Rd, pd, vd



def geodesic(w0, w1, p0, p1, b):
    return np.sqrt(b * np.linalg.norm(w0 - w1) ** 2 + np.linalg.norm(p0 - p1) ** 2)

@jit(nopython=True)
def se3_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3, 3] = - (T[:3,:3].T @ T[:3,3]) 
    return T_inv

def random_sample_ball(n_dim: int=3, n_samples: int=1, radius: float=1):
    # See this paper: https://arxiv.org/pdf/math/0503650
    # sampled from e^-t
    Z = np.log(1/(1 - np.random.rand(n_samples)))
    # sampled from 1/sqrt(pi) e^-t^2
    u = (np.random.rand(n_samples, n_dim) - 0.5)
    # G = np.sign(u) * erfinv(2 * np.sign(u) * u)
    G = erfinv(2 * u)

    # combine according to Theorem 1
    V = radius * G / np.sqrt(Z[:,None] + (G**2).sum(-1))

    return V

@jit(nopython=True) #-- no calculation happening here, so no acceleration needed
def imu_preintegration_and_se3_events(t, R, pos, vel, gyro_i, accel_i,gyro,accel, threshold=0.01, add_polarity = False):
    events = []

    Rk = R[0].copy()
    pk = pos[0].copy()
    vk = vel[0].copy()
    Tk = np.eye(4)
    Tk[:3,:3] = Rk.copy()
    Tk[:3,3:4] = pk.copy().reshape((-1,1))

    T_ref = np.eye(4)
    T_ref[:3,:3] = Rk.copy()
    T_ref[:3,3:4] = pk.copy().reshape((-1,1))

    tk = t[0]

    ## propagate the IMU samples
    for i in range(1, t.shape[0]):
        # find next position and rotation with preintegration
        Rd, pd, vd = _preintegrate_measurement(Rk, pk, vk, gyro_i[i], accel_i[i], t[i] - tk)

        # generate events for se3
        Td = np.eye(4)
        Td[:3,:3] = Rd.copy()
        Td[:3,3:4] = pd.copy().reshape((-1,1))
        T_ref, se3_events = _geodesic_events_se3_vectorized(T_ref, Tk, Td, tk, t[i], threshold, gyro[i-1:i+1], accel[i-1:i+1], add_polarity = add_polarity)
        events.extend(se3_events)



        Rk = Rd
        pk = pd
        vk = vd
        tk = t[i]
        Tk = Td

    return events, tk, Tk

## function to generate event stack
def generate_event_stack(events, window_size=200, se3=False, start_idx=None):
    dim = 6
    mask = np.ones((events.shape[0])).astype(int)
    if se3==True:
        events_meas = events[:, start_idx:start_idx+6].copy()
        ts = events[:, 0].copy()
    else:
        ts = np.array([event[0] for event in events]).reshape((-1))
        events_meas = np.array([event[2] for event in events]).reshape((-1,6))

    ### event stack grid generation
    window_size = int(window_size)
    B = window_size # number of bins to discretize the volume grid
    events_ti = np.linspace(0, B-1, ts.shape[0], endpoint=True).astype(int)

    input = np.zeros((window_size,dim))
    np.add.at(input,events_ti,events_meas*mask.reshape((-1,1)))
    count = np.zeros((window_size,dim))
    np.add.at(count,events_ti,mask.reshape((-1,1)))
    count[count==0] = 1e-6
    return input/count



def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network

def preprocess_event_gen(ts, feat, gt_ori, gt_pos, v0, threshold):
    R = gt_ori.numpy()
    R0 = Rotation.from_quat(np.concatenate([R[:,1:],R[:,0].reshape(-1,1)], axis=1)).as_matrix().astype('float64').copy()
    # ## feat in imu frame - derotate
    ori_q = quaternion.from_float_array(gt_ori.numpy()).copy()
    gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([feat.shape[0], 1]), feat[:,:3].numpy()], axis=1)).copy()
    acce_q = quaternion.from_float_array(np.concatenate([np.zeros([feat.shape[0], 1]), feat[:,-3:].numpy()], axis=1)).copy()

    gyro_i = quaternion.as_float_array(ori_q.conj() * gyro_q * ori_q)[:, 1:].astype('float64').copy()
    accel_i = quaternion.as_float_array(ori_q.conj() * acce_q * ori_q)[:, 1:].astype('float64').copy()
    
    # ## generate events
    events, tk, Tk = imu_preintegration_and_se3_events(ts.reshape((-1,1))[:,0], R0, gt_pos, v0, gyro_i, accel_i, 
                                                       feat[:,:3].numpy().copy(),feat[:,-3:].numpy().copy(), 
                                                       threshold=threshold, add_polarity=True)

    fl_events = np.zeros((2, 13))
    if len(events) == 0:
        ## calculate polarity
        T_0 = np.eye(4)
        T_0[:3,:3] = R0[0].copy()
        T_0[:3,3:4] = gt_pos[0].copy().reshape((-1,1))
        T_ref_1 = np.linalg.inv(T_0) @ Tk
        w_ref_1 = mat_log_se3(T_ref_1)

        polarity = w_ref_1 / np.linalg.norm(w_ref_1)

        fl_events[:,-6:-3] = (T_0[:3,:3] @ polarity[:3])[None,]
        fl_events[:,-3:] = (T_0[:3,:3] @ polarity[3:])[None,]
    else:
        first_events = np.stack(events[:2])
        first_events = first_events[first_events[:, 0].argsort()]
        fl_events[0,-6:] = first_events[0,-6:].copy()
        last_events = np.stack(events[-2:])
        last_events = last_events[last_events[:, 0].argsort()]
        fl_events[1,-6:] = last_events[-1,-6:].copy()
    

    fl_events[0,0] = ts[0]
    fl_events[1,0] = tk

    fl_events[0,1:4] = feat[:,:3][0].numpy().copy()
    fl_events[0,4:7] = feat[:,-3:][0].numpy().copy()
    fl_events[1,1:4] = feat[:,:3][-1].numpy().copy()
    fl_events[1,4:7] = feat[:,-3:][-1].numpy().copy()

    events.extend(fl_events)

    events = np.stack(events)
    events = events[events[:, 0].argsort()]

    features = []
    features.append(generate_event_stack(events, 200, se3=True, start_idx=1))
    
    pol_ev = generate_event_stack(events, 200, se3=True,start_idx = 7)
    features.append(pol_ev/(np.linalg.norm(pol_ev, axis=1).reshape((-1,1))+1e-4))
    features = np.concatenate(features, axis=1)

    return torch.tensor(np.expand_dims(features.astype(np.float32).T, 0))

def run_test(network, data_loader, device, arch, eval_mode=True, mode ='val'):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _, gt_ori, gt_v0, gt_pos, ts) in enumerate(tqdm(data_loader)):
        global _initial_velocity, _input_channel, _contrast_threshold
        if mode == 'test':
            if bid == 0:
                _initial_velocity[:,:2] = gt_v0.numpy().reshape((1,-1))
            feat = preprocess_event_gen(ts.numpy().reshape((-1,)), feat.squeeze().T, gt_ori.squeeze(), gt_pos.squeeze().numpy(), _initial_velocity, _contrast_threshold)
            assert feat.shape[1] == _input_channel

        pred = network(feat.to(device)).cpu().detach().numpy()
        if mode=='test':
            _initial_velocity[:,:2] = pred.copy() 
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')
    global _input_channel, _output_channel

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
    elif mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = False#False#True

    if mode =='test':
        if args.dataset == 'ronin':
            seq_type = GlobSpeedSequence_test
        elif args.dataset == 'ridi':
            from data_ridi import RIDIGlobSpeedSequence
            seq_type = RIDIGlobSpeedSequence
        dataset = StridedSequenceDataset_test(
            seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
            random_shift=random_shift, transform=transforms, 
            shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error, mode = mode)
        # global _input_channel, _output_channel
        _output_channel = dataset.target_dim
        _input_channel = 12
        return dataset

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        from data_ridi import RIDIGlobSpeedSequence
        seq_type = RIDIGlobSpeedSequence
    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,split_per = args.data_split_percentage,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error, mode = mode, 
        contrast_threshold=args.contrast_threshold, add_vel_perturb_range=args.add_vel_perturb_range, 
        polarity_noise_range=args.polarity_noise_range)
 
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(network, train_loader, device, args.arch, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device, args.arch)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in tqdm(range(start_epoch, args.epochs)):#
            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            for batch_id, (feat, targ, _, _, _, _, _, _) in enumerate(tqdm(train_loader)):
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                pred = network(feat)
                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                loss = criterion(pred, targ)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                # step += 1
                # if batch_id == 200:
                #     break
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device, args.arch)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
            else:
                if args.out_dir is not None and osp.isdir(args.out_dir):
                    model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                    torch.save({'model_state_dict': network.state_dict(),
                                'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    print('Model saved to ', model_path)

            if (epoch+1)%20==0:
                model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_last.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id][200:]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int32)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][200, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    print(ind[0], ts_ext[:3], ts[:3])
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test_sequence(args):

    global _contrast_threshold
    _contrast_threshold = args.contrast_threshold

    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.root_dir
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args, mode='test')

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch)

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60

    for data in test_data_list:
        seq_dataset = get_dataset(root_dir, [data], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False)
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int32)

        targets, preds = run_test(network, seq_loader, device, args.arch,eval_mode=True, mode='test')
        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        pos_pred = recon_traj_with_preds(seq_dataset, preds)
        pos_gt = seq_dataset.gt_pos[0][200:]

        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        print(pos_pred.shape,pos_gt.shape)
        ate, rte = compute_ate_rte(pos_pred[:,:2], pos_gt[:,:2], pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred[:,:2] - pos_gt[:,:2], axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

        # Plot figures
        kp = preds.shape[1]
        if kp == 2:
            targ_names = ['vx', 'vy']
        elif kp == 3:
            targ_names = ['vx', 'vy', 'vz']

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        for i in range(kp):
            plt.subplot2grid((kp, 2), (i, 1))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targets[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, data + '_before_traj_recon.npy'),
                np.concatenate([preds, targets], axis=1))
            np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                    np.concatenate([pos_pred, pos_gt], axis=1))
            plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    # Export a csv file
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(test_data_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}, median ATE:{}, median RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all), np.median(ate_all), np.median(rte_all)))
    return losses_avg


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)

def seed_everything(seed=42):#42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    seed_everything()
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, default='lists/list_train.txt')
    parser.add_argument('--val_list', type=str, default="lists/list_val.txt")
    parser.add_argument('--test_list', type=str, default="lists/list_single_seq.txt")
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default="/home/royinakj/ronin_data/all_data", help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default="ev_data/resnet_train_cache_ev_bth1_v05_v3", help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--arch', type=str, default='resnet18')#resnet18
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default="ev_data/test")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)
    parser.add_argument('--contrast_threshold', type=float, default= 0.1) ## contrast threshold for event generation
    parser.add_argument('--add_vel_perturb_range', type=float, default = 0.5) ## noise for training on gt_velocity
    parser.add_argument('--polarity_noise_range', type=float, default = 0.5) ## noise for training on gt_polarity
    parser.add_argument('--data_split_percentage', type=float, default=1.0) # 0.6 to use 30% training data (available is 50% data)
    # 0.4 for 20% training data
    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})


    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')
