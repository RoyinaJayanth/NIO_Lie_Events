"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from utils.math_utils import mat_exp, mat_exp_vec, mat_log, exp_SE3 as mat_exp_se3, log_SE3 as mat_log_se3
from utils.torch_math_utils import so3_exp_map
from utils.from_scipy import compute_euler_from_matrix
from utils.logging import get_logger
from .constants import *
from scipy import interpolate
from scipy.spatial.transform import Rotation
from numba import jit
from scipy.special import erfinv
import math




@jit(nopython=True)
def sinc_robust(x):
    if np.abs(x) < 1e-3:
        return 1
    else:
        return np.sin(x) / x


@jit(nopython=True)
def Jl_SO3(phi):
    """ Left Jacobian of SO(3) """

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    theta = np.linalg.norm(phi)
    Om = hat(phi)
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
    Exp[:3,3:4] = Jl_SO3(v[3:]) @ v[:3,None]
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
    pol_T_ref = T_ref
    for i in range(len(T_refs)):
        T_refs[i] = T_ref @ mat_exp_steps[i]
        dT_refs[i] = T0_inv @ T_refs[i]
        polarities[i,:3] = (pol_T_ref[:3,:3] @ polarity[:3])[None,]
        polarities[i,3:] = (pol_T_ref[:3,:3] @ polarity[3:])[None,]
        pol_T_ref = T_refs[i]

    dT_0_ref_algs = mat_log_se3_vectorized(dT_refs)

    bils = dT_0_ref_algs.dot(n_w01[:,None])[:,0] / np.linalg.norm(dT_0_1_alg)
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
    V_inv = Jl_SO3_inv_vectorized(w).astype("float64")
    result = np.empty((len(T), 6), dtype=np.float64)
    for i in range(len(T)):
        result[i,:3] = V_inv[i] @ T[i,:3,3]
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

    jl_vec = Jl_SO3_vectorized(v[:, 3:])
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



log = get_logger(__name__)


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


@jit(nopython=True)
def _geodesic_events_translation(p_ref, p0, p1, gyro_01, accel_01, t0, t1, threshold, p_clone=None):
    # Imagine a person starting at p_ref, observing p0, p1.
    # This person takes steps of size threshold toward p1 until this is no longer possible
    # Each time, the time is measured according to the projection along p1 - p0.
    # We need to do this because p0 starts at t0, and p1 is at t1
    events = []

    n_p01 = (p1 - p0) / (np.linalg.norm(p1 - p0)+1e-10)
    polarity = (p1 - p_ref) / (np.linalg.norm(p1 - p_ref)+1e-10)

    while np.linalg.norm(p_ref - p1) > threshold:
        # take step in direction of p1
        p_ref += threshold * polarity

        # calculate linear factor between points t0 and t1
        #bil = ((p_ref - p0).dot(n_p01) / (np.linalg.norm(p_ref - p0)+1e-10))
        bil = ((p_ref - p0).dot(n_p01) / (np.linalg.norm(p1 - p0)+1e-10))

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)

        # gather event
        event = np.zeros((11,))
        event[0] = t_i
        event[1:4] = polarity * threshold
        event[-1] = 1

        gyro_i = gyro_01[0] + (gyro_01[1] - gyro_01[0]) * bil
        acc_i = accel_01[0] + (accel_01[1] - accel_01[0]) * bil 
        event[4:7] = gyro_i
        event[7:10] = acc_i

        events.append(event)

    return p_ref, events


@jit(nopython=True)
def _geodesic_events_rotation(R_ref, R0, R1, gyro_01, accel_01, t0, t1, threshold, R_clone = None):
    # Imagine a person starting at R_ref, observing R0, R1.
    # This person takes steps of size threshold toward R1 until this is no longer possible
    # Each time, the time is measured according to the projection along the shortest path between R1 and R0.
    # We need to do this because p0 starts at t0, and p1 is at t1
    events = []

    n_w01 = mat_log(R0.T @ R1) / (np.linalg.norm(mat_log(R0.T @ R1))+1e-10)

    while np.linalg.norm(mat_log(R_ref.T @ R1)) > threshold:
        # take step in direction of p1
        w_ref_1 = mat_log(R_ref.T @ R1)
        polarity = w_ref_1 / (np.linalg.norm(w_ref_1)+1e-10)
        step = threshold * polarity
        R_ref = R_ref @ mat_exp(step)

        # calculate linear factor between points t0 and t1
        #bil = mat_log(R0.T @ R_ref).dot(n_w01) / (np.linalg.norm(mat_log(R0.T @ R_ref))+1e-10)
        bil = mat_log(R0.T @ R_ref).dot(n_w01) / (np.linalg.norm(mat_log(R0.T @ R1))+1e-10)

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)
        
        # gather event
        event = np.zeros((11,))
        event[0] = t_i
        event[1:4] = polarity * threshold
        event[-1] = 2
        gyro_i = gyro_01[0] + (gyro_01[1] - gyro_01[0]) * bil
        acc_i = accel_01[0] + (accel_01[1] - accel_01[0]) * bil
        event[4:7] = gyro_i
        event[7:10] = acc_i

        events.append(event)

    return R_ref, events

@jit(nopython=True)
def se3_inv(T):
    #return np.linalg.inv(T)
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    #T[:3, 3] = - (T[:3,0] *T[0, 3] + T[:3,1] *T[1, 3] + T[:3,2] *T[2, 3]) 
    T_inv[:3, 3] = - (T[:3,:3].T @ T[:3,3]) 
    return T_inv

@jit(nopython=True)
def _geodesic_events_se3(T_ref, T0, T1, t0, t1, threshold, gyro_01=None, accel_01=None, add_polarity=False):
    # Imagine starting at T_ref, observing T0, T1.
    # Take steps of size threshold toward T1 until this is no longer possible
    # Each time, the time is measured according to the projection along the shortest path between T1 and T0.
    # We need to do this because T0 starts at t0, and T1 is at t1
    # print(T_ref.data.contiguous, T0.data.contiguous, T1.data.contiguous)

    events = []

    dT_0_1_alg = mat_log_se3(se3_inv(T0) @ T1)
    n_w01 = dT_0_1_alg / np.linalg.norm(dT_0_1_alg)

    w_ref_1 = mat_log_se3(se3_inv(T_ref) @ T1)

    while np.linalg.norm(w_ref_1) > threshold:
        # take step in direction of p1
        #w_ref_1 = mat_log_se3(np.linalg.inv(T_ref) @ T1)
        polarity = w_ref_1 / np.linalg.norm(w_ref_1)
        step = threshold * polarity
        T_ref = T_ref @ mat_exp_se3(step)
        w_ref_1 = mat_log_se3(se3_inv(T_ref) @ T1)
        
        # calculate linear factor between points t0 and t1
        dT_0_ref_alg = mat_log_se3(se3_inv(T0) @ T_ref)
        bil = dT_0_ref_alg.dot(n_w01) / np.linalg.norm(dT_0_ref_alg)

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)

        event_dim = 7
        # if gyro_01 is None:
        #     event_dim = 8
        if add_polarity:
            event_dim += 6

        # event = np.zeros(1).astype(np.float64)
        # event[0] = np.float64(t_i)

        event = np.zeros((event_dim,))
        event[0] = t_i



        if gyro_01 is not None:
            event[1:4] = gyro_01[0] + (gyro_01[1] - gyro_01[0]) * bil
            event[4:7] = accel_01[0] + (accel_01[1] - accel_01[0]) * bil
        else:
            # event[1:5] = Rotation.from_matrix(T_ref[:3,:3]).as_quat(scalar_first=False)
            # event[5:8] = T_ref[:3,3]
            event[1:4] = mat_log(T_ref[:3,:3])
            event[4:7] = T_ref[:3,3]

        if add_polarity:
            # event[-6:] = polarity * threshold
            event[-6:] = polarity

        events.append(event)

    return T_ref, events

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
def imu_preintegration_and_se3_events(t, R, pos, vel, gyro_i, accel_i,gyro,accel, threshold=0.01, 
                                      add_polarity = False, init_ref_random=False, r_random_ball=None, p_random_ball=None):
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

    tk = t[0]#.copy()


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


@jit(nopython=True)
def imu_preintegration_and_geodesic_events(t, R, pos, vel, gyro_i, accel_i,gyro,accel, threshold=0.01, rot_component_weight=2, use_rotation=True,
                                           init_ref_random=False, r_random_ball=None, p_random_ball=None):
    events = []
    all_translation_events = []
    all_rotation_events = []

    Rk = R[0].copy()
    pk = pos[0].copy()
    vk = vel[0].copy()

    R_ref = Rk.copy()
    p_ref = pk.copy()

    tk = t[0]#.copy()

    ## propagate the IMU samples
    for i in range(1, t.shape[0]):
        # find next position and rotation with preintegration
        Rd, pd, vd = _preintegrate_measurement(Rk, pk, vk, gyro_i[i], accel_i[i], t[i] - tk)

        # generate events for translation
        p_ref, translation_events = _geodesic_events_translation(p_ref, pk, pd, gyro[i-1:i+1], accel[i-1:i+1], tk, t[i], threshold, pos[0])
        events.extend(translation_events)
        all_translation_events.extend(translation_events)

        if use_rotation:
            # generate events for rotation
            R_ref, rotation_events = _geodesic_events_rotation(R_ref, Rk, Rd, gyro[i-1:i+1], accel[i-1:i+1], tk, t[i], rot_component_weight * threshold, R[0])
            events.extend(rotation_events)
            all_rotation_events.extend(rotation_events)

        Rk = Rd
        pk = pd
        vk = vd
        tk = t[i]

    return events, all_translation_events, all_rotation_events



class SequencesDataset:
    """
    A template class for sequences dataset in TLIO training.
    Each subclass is expected to load data in a different way, but from the same data format.
    """

    def __init__(
        self,
        data_path,
        split,
        genparams,
        only_n_sequence=-1,
        sequence_subset=None,
        normalize_sensor_data=True,
        verbose=False,
        test_file_path = None,
        event_based_input = False,
        interpolate = False, 
        interp_freq = 200,
        base_freq = 200,
        base_event_stack = False,
        geodesic_event = False, 
        rot_component_weight = 2,
        contrast_threshold = 0.01,
        add_vel_perturb = False,
        add_vel_perturb_range = 0.0,
        se3_events = False,
        polarity_input = False,
        imu_channel_freq = 200.0,
        gyro_bias_range = 0.0,
        accel_bias_range = 0.0,
        theta_range_deg = 0.0,
        polarity_noise_range = 0.0,
        perturb_gravity = False,
        noise_before_event_gen = False,
        gravity_noise_before_event_gen = False,
        init_vel_noise_sens = False,
        arch = 'resnet'

    ):
        self.data_path = data_path
        self.split = split
        self.genparams = genparams
        
        self.only_n_sequence = only_n_sequence
        self.sequence_subset = sequence_subset
        self.normalize_sensor_data = normalize_sensor_data
        self.verbose = verbose
        self.test_file_path = test_file_path
        self.event_based_input = event_based_input
        self.interpolate = interpolate
        self.interp_freq = interp_freq
        self.base_freq = base_freq
        self.base_event_stack = base_event_stack
        self.geodesic_event = geodesic_event
        self.rot_component_weight = rot_component_weight
        self.contrast_threshold = contrast_threshold
        self.add_vel_perturb = add_vel_perturb
        self.add_vel_perturb_range = add_vel_perturb_range
        self.se3_events = se3_events
        self.polarity_input = polarity_input
        self.imu_channel_freq = imu_channel_freq
        self.gyro_bias_range = gyro_bias_range
        self.accel_bias_range = accel_bias_range
        self.theta_range_deg = theta_range_deg
        self.polarity_noise_range = polarity_noise_range
        self.arch = arch
        # The list of relevant sensor file names based on data_style
        self.sensor_file_basenames = self.get_sensor_file_basenames()
        self.perturb_gravity = perturb_gravity

        self.noise_before_event_gen = noise_before_event_gen
        self.gravity_noise_before_event_gen = gravity_noise_before_event_gen
        self.init_vel_noise_sens = init_vel_noise_sens
            
        # Index the mem-mapped files and open them (data is not read from disk here)
        self.load_list()
        if self.verbose:
            self.log_dataset_info()
        
    def get_base_sensor_name(self):
        return self.sensor_file_basenames[0]
    
    def load_list(self):
        assert torch.utils.data.get_worker_info() is None, "load_list() can only be called in main proc!"

        #list_info = np.loadtxt(
        #    os.path.join(self.data_path, f"{self.split}_list.txt"), 
        #    dtype=np.dtype(str),
        #)
        if self.test_file_path is not None:
            with open(os.path.join(self.data_path, self.test_file_path)) as f:
                list_info = np.array([s.strip() for s in f.readlines() if len(s.strip()) > 0])
        else:
            with open(os.path.join(self.data_path, f"{self.split}_list.txt")) as f:
                list_info = np.array([s.strip() for s in f.readlines() if len(s.strip()) > 0])

        # For picking exactly some particular sequences
        if self.sequence_subset is not None:
            to_keep = np.array([s in self.sequence_subset for s in list_info])
            assert np.count_nonzero(to_keep) == len(self.sequence_subset), \
                    f"Could not find some sequences from sequence_subset in data list"
            list_info = list_info[to_keep]

        if self.split == "train" and self.only_n_sequence > 0:
            list_info = list_info[:self.only_n_sequence]
            
        # Handle empty lists (i.e., if you don't want to do test or val or something)
        self.data_list = []
        if len(list_info) > 0:
            self.data_list = list_info
        
        # Load the descriptions of all the data (column info and num rows)
        self.data_descriptions = []
        seqs_to_remove = [] # The seqs, not the index
        for seq_id in self.data_list:
            seq_desc = {}
            valid = True
            for i, sensor_basename in enumerate(self.sensor_file_basenames):
                with open(os.path.join(self.data_path, seq_id, 
                        sensor_basename+"_description.json"), 'r') as f: 
                    d = json.load(f)
                    if i == 0 and d["num_rows"] < self.genparams.window_size:
                        valid = False
                        log.warning(f"Sequence {seq_id} being ignored since it is too short ({d['num_rows']} rows)")
                        break
                    seq_desc[sensor_basename] = d
            
            if valid:
                self.data_descriptions.append(seq_desc)
            else:
                seqs_to_remove.append(seq_id)

        # Remove too short sequences from list
        if len(seqs_to_remove) > 0:
            self.data_list = np.array([seq for seq in self.data_list if seq not in seqs_to_remove])

    def get_sensor_file_basenames(self):
        if self.genparams.data_style == "aligned":
            return [COMBINED_SENSOR_NAME]
        elif self.genparams.data_style == "resampled":
            return [s + "_resampled" for s in ALL_SENSORS_LIST if s in self.genparams.input_sensors]
        elif self.genparams.data_style == "raw":
            return [s for s in ALL_SENSORS_LIST if s in self.genparams.input_sensors]
        else:
            raise ValueError(f"Invalid data_style {self.genparams.data_style}")
    
    def log_dataset_info(self):
        cumulated_duration_hrs = 0
        self.max_num_rows = None
        self.min_num_rows = None
        for i, seq_id in enumerate(self.data_list):
            seq_fps = {}
            desc = self.data_descriptions[i]
            for j, sensor_basename in enumerate(self.sensor_file_basenames):
                sensor_desc = desc[sensor_basename]
                num_cols = sum([
                    int(c.split("(")[1].split(")")[0]) for c in sensor_desc["columns_name(width)"]
                ])
                cumulated_duration_hrs += 1e-6 * (sensor_desc["t_end_us"] - sensor_desc["t_start_us"]) / 60 / 60
                self.max_num_rows = (
                    sensor_desc["num_rows"] if self.max_num_rows is None
                    else max(sensor_desc["num_rows"], self.max_num_rows)
                )
                self.min_num_rows = (
                    sensor_desc["num_rows"] if self.min_num_rows is None
                    else min(sensor_desc["num_rows"], self.min_num_rows)
                )
    
        # log some statitstics
        #log.info(f"Using these sequences: {list(self.data_list)}")
        log.info(
            f"Cumulated {self.split} dataset duration is {cumulated_duration_hrs:.3f} hours"
        )
        log.info(
            f"Number of {self.split} sequences is {len(self.data_descriptions)}"
        )
        #log.info(
        #    f"Number of {self.split} samples is {self.length} "
        #    f"(decimated by {self.genparams.decimator}x)"
        #)
        log.info(f"Min/max sequences length={self.min_num_rows}, {self.max_num_rows}") 
    
    def poses_to_target(self, rot, pos):
        # Calculate relative info on the fly
        # targ is what we want to regress from these features
        R_W_0 = Rotation.from_quat(rot[0:1]).as_matrix()
        R_W_i = Rotation.from_quat(rot).as_matrix()

        # NOTE R_W_i @ R_W_0.transpose() looks strange, but it is the delta rotation between the two times
        # aligned with the world frame instead of body frame.
        targ_dR_World = R_W_i @ R_W_0.transpose([0,2,1])
        targ_dt_World = pos - pos[0:1] # Displacement in global frame
        return targ_dR_World, targ_dt_World
    
    def positional_encoding(self, x, num_frequencies, incl_input=True):
        prepend = 1 if incl_input else 0
        enc_sz = x.shape[1] * (prepend + 2 * num_frequencies)
        res = np.zeros((x.shape[0], enc_sz))

        if incl_input:
            res[:, :x.shape[1]] = x
            #np.sin(x/np.power(1e4, (2*i)/x.shape[0]))
        powers = np.power(1e4, (2*np.arange(num_frequencies))/x.shape[0]) # (L,)
        sin_phases =  x[:, None, :]/powers[None, :, None] # (N, L, D)
        cos_phases = np.pi / 2 - sin_phases
        phases = np.stack([sin_phases, cos_phases], axis=-2) # (N, L, 2, D)
        # print(phases.shape)
        flat = phases.reshape((phases.shape[0], -1))
        res[:, prepend*x.shape[1]:] = np.sin(flat)

        return res
    

    ## function to generate event stack
    def generate_event_stack(self, events, window_size=200, new=False, se3=False, start_idx=None, polarity_id = None):
        dim = 6
        mask = np.ones((events.shape[0])).astype(int)
        if polarity_id is not None:
            dim=3
            ts = events[:, 0].copy()
            events_meas = events[:,start_idx:start_idx+3].copy()
            mask = polarity_id.copy()
        elif se3==True:
            events_meas = events[:, start_idx:start_idx+6].copy()
            ts = events[:, 0].copy()
        elif start_idx is not None:
            dim = 3
            events_meas = events[:, start_idx:start_idx+3].copy()
            ts = events[:, 0].copy()
        elif new==True:
            events_meas = events[:, 4:].copy()
            ts = events[:, 0].copy()
        
        else:
            ts = np.array([event[0] for event in events]).reshape((-1))
            p = np.array([event[1] for event in events]).reshape((-1))
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
    
    def gen_base_event_stack(self, ts, feats_imu0):
        B = 200 # number of bins to discretize the volume grid
        events_ti = np.linspace(0, B-1, ts.shape[0], endpoint=True).astype(int)
        
        input = np.zeros((200,6))
        np.add.at(input,events_ti,feats_imu0)
        count = np.zeros((200,6))
        np.add.at(count,events_ti,int(1))
        count[count==0] = 1e-6
        return input/count


    def unpack_data_window(self, seq_data, seq_desc, row):
        feats = {}
        ts_us_base_sensor = None
        if self.genparams.data_style == "aligned":
            data_chunk = seq_data[COMBINED_SENSOR_NAME][row:row+self.genparams.window_size]
            # Make sure idx was valid with sufficient padding for window
            assert data_chunk.shape[0] == self.genparams.window_size
            
            #ts_us, gyr, acc, rot, pos, vel = np.hsplit(data_chunk, [1, 4, 7, 11, 14])
            ts_us = data_chunk[:,0:1]
            ts_us_base_sensor = np.copy(ts_us)
            ## adding positional encoded timestamp as a feature (needed for eq_transformer arch)
            ts_ind = np.arange(1, ts_us.shape[0]+1)
            feats["pe_ts"] = self.positional_encoding(ts_ind.reshape((-1,1)), num_frequencies=4, incl_input=True)

            # Check which sensor data we need to load
            # TODO make this less hard-coded for the columns somehow
            # Maybe the description json could be better for getting the columns
            #if "imu0" in self.genparams.input_sensors:
            if "imu0" in self.genparams.input_sensors:
                feats["imu0"] = data_chunk[:,1:7] # gyro0, accelerometer0
            if "imu1" in self.genparams.input_sensors:
                feats["imu1"] = data_chunk[:,7:13] # gyro1, accelerometer1
            if "mag0" in self.genparams.input_sensors:
                feats["mag0"] = data_chunk[:,13:16]
            if "barom0" in self.genparams.input_sensors:
                feats["barom0"] = data_chunk[:,16:18]
            
            # All sensors have the same timestamps in this data_style, just concat here
            #ts_normalized = 2 * (ts_us - ts_us[0]) / (ts_us[-1] - ts_us[0]) - 1
            for k, v in feats.items():
            #    feats[k] = np.concatenate([ts_normalized, v], axis=1).astype(np.float64).T
                feats[k] = v.astype(np.float64).T
            
            rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]

        elif self.genparams.data_style == "resampled":
            
            # With resampled data, the "approximate_frequency" in the json file is exact,
            # so we can quickly index the timestamps of sensors in different memmap files.
            base_sensor_freq = seq_desc[self.get_base_sensor_name()]["approximate_frequency_hz"]
            base_sensor_window_start_time = None
            base_sensor_window_end_time = None
            for i, sensor_name in enumerate(self.sensor_file_basenames):
                if i == 0:
                    sensor_row = row
                    window_size = self.genparams.window_size
                else:
                    # Index the row based on sensor frequency.
                    sensor_freq = seq_desc[sensor_name]["approximate_frequency_hz"]
                    sensor_seq_start_time = seq_desc[sensor_name]["t_start_us"]
                    # TODO off by one possible here from rounding/flooring
                    sensor_row = int(1e-6*(base_sensor_window_start_time - sensor_seq_start_time) * sensor_freq)
                    # TODO should calculate all the window sizes at startup so that we don't
                    # accidentally get an off-by-one window size error from float errors
                    window_size = int(self.genparams.window_size * sensor_freq / base_sensor_freq)
            
                
                data_chunk = seq_data[sensor_name][sensor_row:sensor_row+window_size]
                ts_us = data_chunk[:,0:1]
                ts_us_base_sensor = np.copy(ts_us)
                rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]

                gt_data = ts_us_base_sensor, rot, pos.astype(np.float64), vel.astype(np.float64)

                # if "imu" in sensor_name:
                #     feat = data_chunk[:,1:7] # gyro, accelerometer
                if "mag" in sensor_name:
                    feat = data_chunk[:,1:4]
                if "barom" in sensor_name:
                    feat = data_chunk[:,1:3]

                # Make sure idx was valid with sufficient padding for window
                assert data_chunk.shape[0] == window_size
                # del data_chunk

                if self.interpolate or (self.base_freq!=self.interp_freq) or (self.base_event_stack and self.split=='train') or (self.imu_channel_freq!=self.interp_freq):
                    data_chunk = seq_data[sensor_name][sensor_row-1:sensor_row+window_size]
                    if "imu" in sensor_name:
                        feat = data_chunk[:,1:7] # gyro, accelerometer
                        ts_us_base_sensor = data_chunk[:,0:1]
                        rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]
                
                    assert data_chunk.shape[0] == window_size+1
                    assert ts_us.shape[0] == window_size
                else:
                    if "imu" in sensor_name:
                        feat = data_chunk[:,1:7] # gyro, accelerometer
                
                ts_ind = np.arange(1, ts_us.shape[0]+1)
                feats["pe_ts"] = self.positional_encoding(ts_ind.reshape((-1,1)), num_frequencies=4, incl_input=True)
                if i == 0:
                    
                    base_sensor_window_start_time = ts_us[0]
                    base_sensor_window_end_time = ts_us[-1]
                    # GT data comes from base sensor
                    
                
                
          
      
                feats[sensor_name.split("_")[0]] = feat.T
                stored_ts_us = np.copy(ts_us_base_sensor)
                stored_feat = np.copy(feat)
                stored_R = Rotation.from_quat(rot).as_matrix()
                stored_pos = np.copy(pos)

        else:
            raise NotImplementedError()
        
        if self.se3_events and self.split=='train' and self.accel_bias_range!=0.0:
        

        #     # bias augmentation
            assert feats['imu0'].shape[0] == 6
            # shift in the accel and gyro bias terms
            feats['imu0'][:3, :] += (
                (np.random.randn(3, 1) - 0.5)
                * self.gyro_bias_range / 0.5
            )
            feats['imu0'][3:6, :] += (
                (np.random.randn(3, 1) - 0.5)
                * self.accel_bias_range / 0.5
            )
        if self.se3_events and self.split=='train' and self.theta_range_deg!=0.0 and self.perturb_gravity==False:
        #     ## gravity perturbation - applied only to imu data
            angle_rand_rad = np.random.rand(1)[0] * math.pi * 2
            theta_rand_rad = np.random.rand(1)[0]* math.pi* self.theta_range_deg/ 180.0
            c = np.cos(angle_rand_rad)
            s = np.sin(angle_rand_rad)
            vec_rand = np.array([c, s, 0]).reshape((1,3))
            rvec = theta_rand_rad * vec_rand  # N x 3
            R_mat = so3_exp_map(torch.from_numpy(rvec))  # N x 3 x 3
            R_mat = R_mat.numpy()[0]

            rot = Rotation.from_quat(rot).as_matrix()
            rot = np.einsum("ij,tjk->tik", R_mat, rot)
            rot = Rotation.from_matrix(rot).as_quat().astype(np.float64)


            pos = np.einsum("ij,tj->ti", R_mat, pos)
            vel = np.einsum("ij,tj->ti", R_mat, vel)

            ## not changing the ground truth
            # gt_data = ts_us_base_sensor, rot, pos.astype(np.float64), vel.astype(np.float64)

            # Only IMU and mag data need to be rotated (not barometer) - since the input is compensated rotated in world from npy file
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is also being aligned to local gravity aligned!')
                    assert feats[k].shape[0] == 6
                    feats[k][:3] = np.einsum("ij,jt->it", R_mat, feats[k][:3])
                    feats[k][3:6] = np.einsum("ij,jt->it", R_mat, feats[k][3:6])                        
                        
                elif "mag" in k:
                    assert feats[k].shape[0] == 3
                    feats[k] = np.einsum("ij,jt->it", R_mat, feats[k])

            

        
        
        if self.split=='train' or self.init_vel_noise_sens:
            add_vel_perturb = self.add_vel_perturb or self.init_vel_noise_sens
        else:
            add_vel_perturb = False

        if self.base_freq == self.interp_freq and self.interp_freq==self.imu_channel_freq:
            self.interpolate = False
            subsample=False
        else:
            subsample = True
        
        if self.interpolate or subsample:
            ts_us_base_sensor  = ts_us_base_sensor[::int(self.base_freq/self.interp_freq)]
            feats[sensor_name.split("_")[0]] = feats['imu0'][:,::int(self.base_freq/self.interp_freq)]
            rot = rot[::int(self.base_freq/self.interp_freq)]
            assert ts_us[-1] == ts_us_base_sensor[-1]



        if self.interpolate and self.event_based_input==False and self.base_event_stack==False:

            ## checking interpolation in world frame - because in local frames the frame continuously changes
            ## interpolating in world frame
            old_ts  = ts_us_base_sensor
            ts = np.zeros((int(self.imu_channel_freq),old_ts.shape[-1]))
            old_feat = feats['imu0']
            feats_new = np.zeros((int(self.imu_channel_freq),old_feat.shape[0])) 
            ## directly interpolate
            ts = np.linspace(ts_us[0], ts_us[-1], int(self.imu_channel_freq), endpoint=True).reshape((-1,1)) ## interpolating always to 200
            # print(old_ts[i,-1], ts[i,-1], imu_freq)
            assert ts_us[-1] == ts[-1]
            assert ts_us[0] == ts[0]
            feats_new = interpolate.interp1d(old_ts.reshape((-1)), old_feat.T, axis=0)(ts.reshape((-1)))

            feats[sensor_name.split("_")[0]] = feats_new.astype(np.float64).T
            


        if self.base_event_stack:

            ts = ts_us_base_sensor*1e-6
            dt = ts[1:] - ts[:-1]
            # repeating first value
            dt = np.concatenate((np.repeat(dt[0], 1).reshape((-1,1)), dt))
            feats_imu0 = feats['imu0'].T.copy()
            events=[]
            
            ## do scale aug while training
            if self.split=='train':
                ## discrete imu frequencies
                freq = [40,50,100,200]#[100,200,160,400,800][100,125,200,250,500,1000][40,50,100,200]
                aug_freq = freq[torch.randperm(4)[0]]
                ts_us_base_sensor  = ts_us_base_sensor[::int(self.base_freq/aug_freq)]
                feats_imu0 = feats_imu0[::int(self.base_freq/aug_freq)]


            feats['imu0'] = self.gen_base_event_stack(ts_us_base_sensor,feats_imu0).T.astype(np.float32)   


        elif self.event_based_input:
            if self.se3_events:
                ts = ts_us_base_sensor*1e-6
                R = Rotation.from_quat(rot).as_matrix()
                pos = np.array(pos)
                vel = np.array(vel)
                

                if add_vel_perturb:
                    vel = vel + (
                    (np.random.uniform(low=0, high=1, size=vel.shape[0]).reshape(-1,1) - 0.5)
                    * self.add_vel_perturb_range / 0.5
                )

                gyro = np.array(feats['imu0'][:3, :].T.copy())
                accel = np.array(feats['imu0'][-3:, :].T.copy())
                gyro_i = np.einsum('tji,tj->ti', R, gyro)
                accel_i = np.einsum('tji,tj->ti', R, accel) 

                
                ## adding bias noise to the imu samples for complementary filter before event generation
                if self.noise_before_event_gen:
                    gyro_i = gyro_i + (
                        (np.random.uniform(low=0, high=1, size=3).reshape(1,-1) - 0.5)
                        * (self.gyro_bias_range/2) / 0.5)
                    accel_i = accel_i + (
                        (np.random.uniform(low=0, high=1, size=3).reshape(1,-1) - 0.5)
                        * (self.accel_bias_range/2) / 0.5)
                    
                ## adding gravity perturbation to imu sample for complementary filter before event generation
                if self.gravity_noise_before_event_gen:
                    ## orientation, position, velocity, imu samples but not output
                    ## gravity perturbation
                    angle_rand_rad = ((np.random.uniform(low=0, high=1, size=1)[0] - 0.5)/0.5) * math.pi * 2
                    theta_rand_rad = ((np.random.uniform(low=0, high=1, size=1)[0] - 0.5)/0.5)* math.pi* self.theta_range_deg/ 180.0
                    c = np.cos(angle_rand_rad)
                    s = np.sin(angle_rand_rad)
                    vec_rand = np.array([c, s, 0]).reshape((1,3))
                    rvec = theta_rand_rad * vec_rand  # N x 3
                    R_mat = so3_exp_map(torch.from_numpy(rvec))  # N x 3 x 3
                    R_mat = R_mat.numpy()[0]

                    gyro_i = np.einsum("ik,tk->ti", R_mat, gyro_i)
                    accel_i = np.einsum("ik,tk->ti", R_mat, accel_i)

                    R=np.einsum("ik,tkj->tij", R_mat, R)
                    pos=np.einsum("ik,tk->ti", R_mat, pos)
                    vel=np.einsum("ik,tk->ti", R_mat, vel)

                    stored_R = np.einsum("ik,tkj->tij", R_mat, stored_R)
                    stored_pos = np.einsum("ik,tk->ti", R_mat, stored_pos)

                r_random_ball = random_sample_ball(radius=self.polarity_noise_range)[0]
                p_random_ball = random_sample_ball(radius=self.polarity_noise_range)[0]

                if self.polarity_input:
                    events, tk, Tk = imu_preintegration_and_se3_events(ts[:,0], R, pos, vel, gyro_i, accel_i, gyro,accel, 
                                                            threshold=self.contrast_threshold,
                                                            add_polarity=self.polarity_input, init_ref_random=False, 
                                                            r_random_ball=r_random_ball, p_random_ball=p_random_ball)
                    
                    
                    fl_events = np.zeros((2, 13))
                    ## for aria and different rate inputs
                    index_fl = stored_ts_us.tolist().index(ts_us[0])
                    if len(events) == 0:
                        ## calculate polarity
                        T_0 = np.eye(4)
                        T_0[:3,:3] = stored_R[index_fl].copy()
                        T_0[:3,3:4] = stored_pos[index_fl].copy().reshape((-1,1))
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
                    

                    fl_events[0,0] = stored_ts_us[index_fl]*1e-6
                    fl_events[1,0] = tk

                    events.extend(fl_events)

                    if len(events) == 0:
                        events = np.zeros((1, 13))
                    else:
                        events = np.stack(events)
                        events = events[events[:, 0].argsort()]
                        ## make sure we don't have events before the time
                        index_fl = np.where(events[:,0]>=(ts_us[0]*1e-6))[0].min()
                        events = events[index_fl:]
                    features = []
                    features.append(self.generate_event_stack(events, ts, window_size = self.imu_channel_freq, se3=True, start_idx=1).T.astype(np.float32))
                    
                    if self.gravity_noise_before_event_gen:
                        ## put back polarity
                        events[:,7:10] = np.einsum("ki,tk->ti", R_mat, events[:,7:10])
                        events[:,10:] = np.einsum("ki,tk->ti", R_mat, events[:,10:])


                    ## adding additive noise to polarity
                    events[:,7:] = events[:,7:] + (
                        (np.random.uniform(low=0, high=1, size=6).reshape(1,-1) - 0.5)
                        * self.polarity_noise_range / 0.5)
                    pol_ev = self.generate_event_stack(events, ts, window_size = self.imu_channel_freq, se3=True,start_idx = 7).T.astype(np.float32)
                    features.append(pol_ev/(np.linalg.norm(pol_ev, axis=0).reshape((1,-1))+1e-4))
                    feats['imu0'] = np.concatenate(features, axis=0)
                else:
                    events, _, Tk = imu_preintegration_and_se3_events(ts[:,0], R, pos, vel, gyro_i, accel_i, gyro,accel, 
                                                            threshold=self.contrast_threshold, init_ref_random=False, 
                                                            r_random_ball=r_random_ball, p_random_ball=p_random_ball)
                    if len(events) == 0:
                        events = np.zeros((40, 7))
                    else:
                        events = np.stack(events)
                        events = events[events[:, 0].argsort()]

                    
                    events = np.concatenate([
                        np.concatenate([ts[0].reshape((1,1)), np.array(feats['imu0'].T[0]).reshape((1,-1))], axis=-1).reshape((1,-1)),
                        events,
                        np.concatenate([ts[-1].reshape((1,1)), np.array(feats['imu0'].T[-1]).reshape((1,-1))], axis=-1).reshape((1,-1))
                    ])
                    
                    feats['imu0'] = self.generate_event_stack(events, ts, window_size = self.imu_channel_freq, se3=True, start_idx=1).T.astype(np.float32)

            else:
                ts = ts_us_base_sensor*1e-6
                R = Rotation.from_quat(rot).as_matrix()
                pos = np.array(pos)
                vel = np.array(vel)

                
                if add_vel_perturb:
                    vel = vel + (
                    (np.random.uniform(low=0, high=1, size=vel.shape[0]).reshape(-1,1) - 0.5)
                    * self.add_vel_perturb_range / 0.5
                )

                gyro = np.array(feats['imu0'][:3, :].T.copy())
                accel = np.array(feats['imu0'][-3:, :].T.copy())
                gyro_i = np.einsum('tji,tj->ti', R, gyro)
                accel_i = np.einsum('tji,tj->ti', R, accel) 

                r_random_ball = random_sample_ball(radius=0.01)[0]
                p_random_ball = random_sample_ball(radius=0.01)[0]

                events,  translation_events, rotation_events = imu_preintegration_and_geodesic_events(ts[:,0], R, pos, vel, gyro_i, accel_i, gyro,accel, threshold=self.contrast_threshold,
                                                                rot_component_weight=self.rot_component_weight, use_rotation=self.geodesic_event, init_ref_random=False,
                                                                p_random_ball=p_random_ball, r_random_ball=r_random_ball)
                if len(events) == 0:
                    events = np.zeros((1, 10))
                    dt = np.zeros((1,1))
                else:
                    events = np.stack(events)
                    events = events[events[:, 0].argsort()]
                    events = events[:, :-1]
                    assert events.shape[1] == 10
                    dt = np.append(ts[0], events[:,0],0)
                    dt = dt[1:] - dt[:-1]
                    dt = dt.reshape((-1,1))
                    

                
                events = np.concatenate([
                    np.concatenate([np.array([ts[0,0], 0, 0 , 1]), np.array(feats['imu0'].T[0])]).reshape((1,-1)),
                    events,
                    np.concatenate([np.array([ts[-1,0], 0, 0 , 1]), np.array(feats['imu0'].T[-1])]).reshape((1,-1))
                ])
                dt = np.concatenate([np.array([ts[1,0]-ts[0,0]]).reshape((1,1)), dt, np.array([ts[-1,0]-ts[-2,0]]).reshape((1,1))])
                
                
                feats['imu0'] = self.generate_event_stack(events, ts, window_size = self.imu_channel_freq, new=True).T.astype(np.float32)
            
        else:
            events = []
           
        # if len(events):
        #     print(len(events))
        feats['imu0'] = feats['imu0'].astype(np.float32)

        return feats, gt_data,len(events)

    def data_chunk_from_seq_data(self, seq_data, seq_desc, row):
        
        feats, gt_data, no_events = self.unpack_data_window(seq_data, seq_desc, row)

        # Normalize the raw sensor data into something better for learning (sensor-dependent)
        if self.normalize_sensor_data:
            feats = self.normalize_feats(feats)
        
        ## flag to experiment with and without gravity compensating accel data -- we can directly subtract as the data is in world frame
        if self.genparams.g_compensate:
            # print('entering gravity compensation')
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is being gravity compensated!')
                    assert feats[k].shape[0] == 6
                    feats[k] -= np.array([0, 0, 0, 0, 0, 9.81]).reshape((-1,1)) # feat = gyr, accel and shape is (6, samples)
            
        

        ts_us, rot, pos, vel = gt_data
        targ_dR_World, targ_dt_World = self.poses_to_target(rot, pos)

        rot = Rotation.from_quat(rot).as_matrix()

        R_world_gla = np.eye(3)
        if self.genparams.express_in_local_gravity_aligned:# and self.polarity_input==False
            #assert False
            # print('Output data is also being aligned to local gravity aligned!')
            R_W_0 = rot[0:1]
            angles_t0 = compute_euler_from_matrix(
                R_W_0, "xyz", extrinsic=True
            )
            ri_z = angles_t0[0,2]
            c = np.cos(ri_z)
            s = np.sin(ri_z)
            R_world_gla = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            targ_dt_World = np.einsum("ji,tj->ti", R_world_gla, targ_dt_World)
            rot = np.einsum("ji,tjk->tik", R_world_gla, rot)
            pos = np.einsum("ji,tj->ti", R_world_gla, pos)
            vel = np.einsum("ji,tj->ti", R_world_gla, vel)

            # Only IMU and mag data need to be rotated (not barometer) - since the input is compensated rotated in world from npy file
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is also being aligned to local gravity aligned!')
                    feats[k][:3] = np.einsum("ji,jt->it", R_world_gla, feats[k][:3])
                    feats[k][3:6] = np.einsum("ji,jt->it", R_world_gla, feats[k][3:6])
                    if self.polarity_input:
                        assert feats[k].shape[0] == 12
                        feats[k][6:9] = np.einsum("ji,jt->it", R_world_gla, feats[k][6:9])
                        feats[k][9:12] = np.einsum("ji,jt->it", R_world_gla, feats[k][9:12])
                    else:
                        assert feats[k].shape[0] == 6
                        
                elif "mag" in k:
                    assert feats[k].shape[0] == 3
                    feats[k] = np.einsum("ji,jt->it", R_world_gla, feats[k])
            
        elif self.genparams.express_in_local_frame: 
            #assert False
            # print('Entered local frame data prep step!')
            R_world_gla = rot[0]
            #print(Rotation.from_quat(rot[0]).as_matrix().shape)
            targ_dt_World = np.einsum("ji,tj->ti", R_world_gla, targ_dt_World)
            # Only IMU and mag data need to be rotated (not barometer) - since the input is compensated rotated in world from npy file
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is also being aligned to local frame!')
                    assert feats[k].shape[0] == 6
                    feats[k][:3] = np.einsum("ji,jt->it", R_world_gla, feats[k][:3])
                    feats[k][3:] = np.einsum("ji,jt->it", R_world_gla, feats[k][3:])
                elif "mag" in k:
                    assert feats[k].shape[0] == 3
                    feats[k] = np.einsum("ji,jt->it", R_world_gla, feats[k])
           
            

        else: ## global frame

            ## since both input and output are already in global frame, no rotations needed
            # print('global frame! No rotations for both imu and output data!')
            targ_dt_World = targ_dt_World
            feats = feats
        feats['feat_o2'] = self.preprocess_o2(feats['imu0'])
        # We may return multiple windows, so place them all in here for convenience.

        

        windows = {
            "main": {
                "ts_us": ts_us,
                "feats": feats,
                "targ_dR_World": targ_dR_World.astype(np.float32),
                "targ_dt_World": targ_dt_World.astype(np.float32),
                "vel_World": vel.astype(np.float32),
                "R_world_gla": R_world_gla,
                "rot" : rot,
                "pos" : pos,
                "no_events": no_events,
            }
        }

        return seq_desc[self.get_base_sensor_name()], windows
    
    def normalize_feats(self, feats):
        """
        Normalize the sensor data from its raw form to some normalized form, typically in [-1,1] or [0,1].
        """
        
        new_feats = {}
        for sensor_name, feat in feats.items():
            # Note that all feat are [1+C,T] where C is channels in sensor data and T is tme dimension.
            # The 1+ is because the sensor data is concatenated with normalized time stamp.
            new_feat = np.copy(feat)
            # Check for nan/inf here (sometimes can pop up in the data)
            new_feat[~np.isfinite(new_feat)] = 0.0
            """  NOTE makes values too small, and disrupts bias perturbation logic
            if "imu" in sensor_name:
                assert new_feat.shape[0] == 6
                # See T74692750 for more info.
                # Out of the two IMUs, the one with the max range is at +/-8G and +/-1000 deg/sec.
                # Normalize by this one so that both IMU values have the same meaning, and are normalized in [-1,1]
                minmax_acc_range_g = 8 # In unit of Gs
                minmax_ang_vel_range_deg_per_sec = 1000
                # IMU values should be in [-1,1] after this
                new_feat[:3] = new_feat[:3] / (minmax_ang_vel_range_deg_per_sec / 180 * np.pi) # gyro
                new_feat[3:6] = new_feat[3:6] / (minmax_acc_range_g * 9.81) # accelerometer
            """
            if "mag" in sensor_name:
                assert new_feat.shape[0] == 3
                # Convert to Gauss, which is closer to 1 in magnitude (Earth's field is around .25-.65 Gauss, and 
                # can be negative here since the magnetomete returns a magnetic field vector instead of magnitude)
                GAUSS_IN_TESLA = 10_000
                new_feat[:3] = new_feat[:3] * GAUSS_IN_TESLA
            if "barom" in sensor_name:
                assert new_feat.shape[0] == 2
                # Pressure converted to bar and normalized heuristically to fit into [-1,1] better.
                # Setting -1,1 to be the min/max pressure/temp ever recorded leads to very small differences
                # in the values for normal situations, so just picked min/max based on some normal daily values on Earth.
                PA_IN_BAR = 100_000
                """
                avg_bar = 1.01325 # Average barometric pressure on earth
                max_bar_deviation = 0.01 # plus/minus avg is what we are considering
                min_bar, max_bar = avg_bar - max_bar_deviation, avg_bar + max_bar_deviation
                new_feat[0] = 2 * (new_feat[0] / PA_IN_BAR - min_bar) / (max_bar - min_bar) - 1
                min_temp = -100
                max_temp = 100
                new_feat[1] = 2 * (new_feat[1] - min_temp) / (max_temp - min_temp) - 1
                """
                new_feat[0] /= PA_IN_BAR # convert pa to bar

            new_feats[sensor_name] = new_feat
        
        return new_feats
    
    def load_data_chunk(self, seq_idx, row):
        raise NotImplementedError("Did not override load_data_chunk!!!")

    def load_and_preprocess_data_chunk(self, seq_idx, row_in_seq, num_rows_in_seq):
        # If training, randomize the row a bit so that we can get better coverage of the data
        # while still respecting the decimator and indexing.
        if self.split == "train":
            row_in_seq = min(num_rows_in_seq-1, row_in_seq + np.random.randint(self.genparams.decimator))
        meta_dict, windows = self.load_data_chunk(seq_idx, row_in_seq)

        ret = {
            "seq_id": self.data_list[seq_idx],
        }
        ret.update(windows["main"]) # Main target and GT data corresponding to seq_idx and row_in_seq
        return ret

    ##########################################################################
    # Functions needed by IntegrateRoninCallback
    ##########################################################################

    def get_ts_last_imu_us(self, seq_idx=0):
        raise NotImplementedError("Did not override get_ts_last_imu_us!!!")

    def get_gt_traj_center_window_times(self, seq_idx=0):
        raise NotImplementedError("Did not override get_gt_traj_center_window_times!!!")
