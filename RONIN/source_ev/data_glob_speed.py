import json
import random
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences

from utils_tlio.math_utils import mat_exp, mat_log, exp_SE3 as mat_exp_se3, log_SE3 as mat_log_se3 # type: ignore

from numba import jit
from scipy.special import erfinv
from scipy.spatial.transform import Rotation
import time


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
    x = 0.5 * (np.trace(R) - 1)  # np.clip(0.5*(np.trace(R) - 1), -1, 1)
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

class GlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 12
    target_dim = 2
    aux_dim = 8
    add_vel_perturb_range = 0.5
    contrast_threshold = 0.1
    polarity_noise_range = 0.5

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        self.mode = kwargs.get('mode','train')
        self.contrast_threshold = kwargs.get('contrast_threshold',self.contrast_threshold)
        self.add_vel_perturb_range = kwargs.get('add_vel_perturb_range', self.add_vel_perturb_range)
        self.polarity_noise_range = kwargs.get('polarity_noise_range', self.polarity_noise_range)
        if data_path is not None:
            self.load(data_path)

        

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
            data_path, self.max_ori_error, self.grv_only)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            gyro_uncalib = f['synced/gyro_uncalib']
            acce_uncalib = f['synced/acce']
            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
            acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))
            ts = np.copy(f['synced/time'])
            tango_pos = np.copy(f['pose/tango_pos'])
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        ## generate events and store event stack as features
        ## initial velocity additive noise
        v0 = glob_v.copy()
        if self.mode == 'train':
            v0 = v0 + (
            (np.random.uniform(low=0, high=1, size=glob_v.shape) - 0.5)
            * self.add_vel_perturb_range / 0.5)
        v0[:,-1] = 0


        R = quaternion.as_float_array(ori_q)
        R0 = Rotation.from_quat(np.concatenate([R[:,1:],R[:,0].reshape(-1,1)], axis=1)).as_matrix()

        start_frame = self.info.get('start_frame', 0)

        step_size = 10
        all_ev_features = []
        all_ts = []
        all_glob_v = []
        all_ori = []
        all_pos = []
        start_time = time.time()
        for ind in range(start_frame, len(glob_v), step_size):
            events, tk, Tk = imu_preintegration_and_se3_events(ts[ind:ind+200].reshape((-1,1))[:,0].copy(), 
                                                               R0[ind:ind+200].copy(), 
                                                               tango_pos[ind:ind+200].copy(), 
                                                               v0[ind-200:ind].copy(), 
                                                               gyro[ind:ind+200].copy(), 
                                                               acce[ind:ind+200].copy(), 
                                                               glob_gyro[ind:ind+200].copy(), 
                                                               glob_acce[ind:ind+200].copy(), 
                                                               threshold=self.contrast_threshold, 
                                                               add_polarity=True)
            ## first and last events
            fl_events = np.zeros((2, 13))
            if len(events) == 0:
                ## calculate polarity
                T_0 = np.eye(4)
                T_0[:3,:3] = R0[ind].copy()
                T_0[:3,3:4] = tango_pos[ind].copy().reshape((-1,1))
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
            

            fl_events[0,0] = ts[ind]
            fl_events[1,0] = tk

            fl_events[0,1:4] = glob_gyro[ind].copy()
            fl_events[0,4:7] = glob_acce[ind].copy()
            fl_events[1,1:4] = glob_gyro[ind+200-1].copy()
            fl_events[1,4:7] = glob_acce[ind+200-1].copy()

            events.extend(fl_events)

            if len(events) == 0:
                events = np.zeros((1, 13))
            else:
                events = np.stack(events)
                events = events[events[:, 0].argsort()]
            # ## polarity noise
            events[:,7:] = events[:,7:] + (
                            (np.random.uniform(low=0, high=1, size=6).reshape(1,-1) - 0.5)
                            * self.polarity_noise_range / 0.5) #self.polarity_noise_range/0.5)

            
            features = []
            features.append(generate_event_stack(events, ts[ind:ind+200],200, se3=True, start_idx=1))
            
            pol_ev = generate_event_stack(events, 200, se3=True,start_idx = 7)
            features.append(pol_ev/(np.linalg.norm(pol_ev, axis=1).reshape((-1,1))+1e-4))
            features = np.concatenate(features, axis=1)

            all_ev_features.append(features)
            all_ts.append(ts[ind:ind+200])
            all_glob_v.append(glob_v[ind:ind+200])
            all_ori.append(R[ind:ind+200])
            all_pos.append(tango_pos[ind:ind+200])




        end_time = time.time()
        print(f"time for one dataset: {(end_time-start_time):.4f} seconds")
        self.ts = np.concatenate(all_ts, axis=0)
        self.features = np.concatenate(all_ev_features, axis=0)
        self.targets = np.concatenate(all_glob_v, axis=0)[:, :2]
        self.orientations = np.concatenate(all_ori, axis=0)
        self.gt_pos = np.concatenate(all_pos, axis=0)
        

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, mode = 'train', contrast_threshold = 0.1, 
                 add_vel_perturb_range = 0.5, polarity_noise_range = 0.5,split_per=1.0, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)
        self.mode = mode
        self.contrast_threshold = contrast_threshold
        self.add_vel_perturb_range = add_vel_perturb_range
        self.polarity_noise_range = polarity_noise_range
        self.split_per = split_per

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, mode = self.mode, 
            contrast_threshold = self.contrast_threshold, add_vel_perturb_range = self.add_vel_perturb_range,
            polarity_noise_range = self.polarity_noise_range, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            indices_map = [[i, j] for j in range(0, self.targets[i].shape[0], 200)]#step_size
            if self.mode == 'train' and self.split_per <1.0:
                indices = np.random.choice(len(indices_map), size=int(self.split_per*len(indices_map)), replace=False)
                self.index_map += np.array(indices_map)[indices].tolist()
            else:
                self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], 200)]#step_size

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        # if self.random_shift > 0:
        #     frame_id += random.randrange(-self.random_shift, self.random_shift)
        #     frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size].astype('float64').copy()
        targ = self.targets[seq_id][frame_id].astype('float64').copy()
        ts = self.ts[seq_id][frame_id:frame_id + self.window_size].astype('float64').copy()

        
        if self.mode=='train':
            
            ## polarity noise
            ## create a mask
            mask = (feat[:,-6:] != 0).astype(int)
            feat[:,-6:] = feat[:,-6:] + (
                            (np.random.uniform(low=0, high=1, size=6).reshape(1,-1) - 0.5)
                            * self.polarity_noise_range / 0.5
                        )* mask

            features = []
            features.append(feat[:,:-6])
            
            pol_ev = feat[:,-6:]
            features.append(pol_ev/(np.linalg.norm(pol_ev, axis=1).reshape((-1,1))+1e-4))
            features = np.concatenate(features, axis=1)
            feat = features
        


        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id, self.orientations[seq_id][frame_id:frame_id + self.window_size], self.targets[seq_id][frame_id], self.gt_pos[seq_id][frame_id:frame_id+2].reshape((-1,3)).astype('float64'), ts

    def __len__(self):
        return len(self.index_map)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=400,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, 5:8])

            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)
