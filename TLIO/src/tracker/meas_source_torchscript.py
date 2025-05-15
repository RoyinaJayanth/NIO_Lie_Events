import numpy as np
import torch
from network.covariance_parametrization import DiagonalParam
from utils.logging import logging
from network.model_factory import get_model
import copy
from scipy import interpolate

from utils.math_utils import mat_exp, mat_exp_vec, mat_log, exp_SE3 as mat_exp_se3, log_SE3 as mat_log_se3
from utils.from_scipy import compute_euler_from_matrix
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
from numba import jit, njit
from scipy.special import erfinv



@jit(nopython=True)
def vee(w_x):
    return np.array([w_x[2, 1], w_x[0, 2], w_x[1, 0]])




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

#@jit(nopython=True)
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

#@jit(nopython=True)
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

#@jit(nopython=True)
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




@jit(nopython=True)
def preintegrate_measurement(R, p, v, gyr, a, dt):
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
def geodesic_events_translation(p_ref, p0, p1, gyro_01, accel_01, t0, t1, threshold):
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
        bil = (p_ref - p0).dot(n_p01) / (np.linalg.norm(p1 - p0)+1e-10)

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)
        gyro_i = gyro_01[0] + (gyro_01[1] - gyro_01[0]) * bil
        acc_i = accel_01[0] + (accel_01[1] - accel_01[0]) * bil

        # gather event
        event = np.zeros((11,))
        event[0] = t_i
        event[1:4] = polarity * threshold
        event[4:7] = gyro_i
        event[7:10] = acc_i
        event[-1] = 1
        events.append(event)

    return p_ref, events


@jit(nopython=True)
def geodesic_events_rotation(R_ref, R0, R1, gyro_01, accel_01, t0, t1, threshold):
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
        bil = mat_log(R0.T @ R_ref).dot(n_w01) / (np.linalg.norm(mat_log(R0.T @ R1))+1e-10)

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)
        gyro_i = gyro_01[0] + (gyro_01[1] - gyro_01[0]) * bil
        acc_i = accel_01[0] + (accel_01[1] - accel_01[0]) * bil

        # gather event
        event = np.zeros((11,))
        event[0] = t_i
        event[1:4] = polarity * threshold
        event[4:7] = gyro_i
        event[7:10] = acc_i
        event[-1] = 2
        events.append(event)

    return R_ref, events

@jit(nopython=True)
def geodesic_events_se3(T_ref, T0, T1, t0, t1, threshold, gyro_01=None, accel_01=None, add_polarity=False):
    # Imagine starting at T_ref, observing T0, T1.
    # Take steps of size threshold toward T1 until this is no longer possible
    # Each time, the time is measured according to the projection along the shortest path between T1 and T0.
    # We need to do this because T0 starts at t0, and T1 is at t1
    events = []

    dT_0_1 = np.ascontiguousarray(np.linalg.inv(T0) @ T1)
    n_w01 = mat_log_se3(dT_0_1) / np.linalg.norm(mat_log_se3(dT_0_1))

    while np.linalg.norm(np.ascontiguousarray(mat_log_se3(np.linalg.inv(T_ref) @ T1))) > threshold:
        # take step in direction of p1
        w_ref_1 = np.ascontiguousarray(mat_log_se3(np.linalg.inv(T_ref) @ T1))
        polarity = w_ref_1 / np.linalg.norm(w_ref_1)
        step = threshold * polarity
        T_ref = np.ascontiguousarray(T_ref @ mat_exp_se3(step))

        # calculate linear factor between points t0 and t1
        dT_0_ref = np.ascontiguousarray(np.linalg.inv(T0) @ T_ref)
        bil = mat_log_se3(dT_0_ref).dot(n_w01) / np.linalg.norm(mat_log_se3(dT_0_ref))

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

# @jit(nopython=True) #-- no calculation happening here, so no acceleration needed
def imu_preintegration_and_se3_events(t, R, pos, vel, gyro_i, accel_i,gyro,accel, threshold=0.01, add_polarity =False, 
                                      vio_preint = False, R_vio=None, p_vio=None, vel_vio=None):
    events = []

    Rk = R[0].copy()
    pk = pos[0].copy()
    vk = vel[0].copy()
    if vio_preint:
        Rk = R_vio[0].copy()
        pk = p_vio[0].copy()
        vk = vel_vio[0].copy()
    Tk = np.eye(4)
    Tk[:3,:3] = Rk.copy()
    Tk[:3,3:4] = pk.copy().reshape((-1,1))

    T_ref = np.eye(4)
    T_ref[:3,:3] = Rk.copy()
    T_ref[:3,3:4] = pk.copy().reshape((-1,1))

    tk = t[0]#.copy()


    ## propagate the IMU samples
    for i in range(1, t.shape[0]):
        ## debug preintegration with vio GT data
        if vio_preint:
            Rd = R_vio[i]
            pd = p_vio[i]
            vd = vel_vio[i]

        else:
            # find next position and rotation with preintegration
            Rd, pd, vd = preintegrate_measurement(Rk, pk, vk, gyro_i[i], accel_i[i], t[i] - tk)

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

    Rk = R[0].copy()
    pk = pos[0].copy()
    vk = vel[0].copy()

    R_ref = Rk.copy()
    p_ref = pk.copy()

    tk = t[0]#.copy()

    if init_ref_random:
        # randomly perturb tk, and Rk by a threshold length away from their initial value
        R_ref = R_ref @ mat_exp(r_random_ball)
        p_ref = p_ref + p_random_ball

    ## propagate the IMU samples
    for i in range(1, t.shape[0]):
        # find next position and rotation with preintegration
        Rd, pd, vd = preintegrate_measurement(Rk, pk, vk, gyro_i[i], accel_i[i], t[i] - tk)

        # generate events for translation
        p_ref, translation_events = geodesic_events_translation(p_ref, pk, pd, gyro[i-1:i+1], accel[i-1:i+1], tk, t[i], threshold)
        events.extend(translation_events)

        if use_rotation:
            # generate events for rotation
            R_ref, rotation_events = geodesic_events_rotation(R_ref, Rk, Rd, gyro[i-1:i+1], accel[i-1:i+1], tk, t[i], rot_component_weight * threshold)
            events.extend(rotation_events)

        Rk = Rd
        pk = pd
        vk = vd
        tk = t[i]

    return events

class MeasSourceTorchScript:
    """ Loading a torchscript has the advantage that we do not need to reconstruct the original network class to
        load the weights, the network structure is contained into the torchscript file.
    """
    def __init__(self, model_path, arch, net_config, force_cpu=False):
        # load trained network model
        logging.info("Loding {}...".format(model_path))
        if not torch.cuda.is_available() or force_cpu:
            torch.init_num_threads()
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = torch.device("cpu")
            # self.net = torch.jit.load(model_path, map_location="cpu")
            
        else:
            self.device = torch.device("cuda:0")
            # NOTE TLIO baseline model won't work on GPU unless we ass map_location
            # https://github.com/pytorch/pytorch/issues/78207
            # self.net = torch.jit.load(model_path, map_location=self.device)#torch.jit.load(model_path, map_location=self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net = get_model(arch, net_config, net_config["input_dim"], net_config["output_dim"]).to(
            self.device
        )
        self.net.load_state_dict(checkpoint["model_state_dict"])

        self.net.to(self.device)
        self.net.eval()
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))
    
    
    def gen_base_event_stack(self, ts, feats_imu0,imu_freq_net):
        B = imu_freq_net # number of bins to discretize the volume grid
        events_ti = np.linspace(0, B-1, ts.shape[0], endpoint=True).astype(int)

        input = np.zeros((imu_freq_net,6))
        np.add.at(input,events_ti,feats_imu0)
        count = np.zeros((imu_freq_net,6))
        np.add.at(count,events_ti,int(1))
        count[count==0] = 1e-6
        return input/count
    
    def generate_event_stack(self, events, imu_freq_net, new=False, se3=False, start_idx=None, polarity_id = None):
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
        imu_freq_net = int(imu_freq_net)
        B = imu_freq_net # number of bins to discretize the volume grid
        events_ti = np.linspace(0, B-1, ts.shape[0], endpoint=True).astype(int)

        input = np.zeros((imu_freq_net,dim))
        np.add.at(input,events_ti,events_meas*mask.reshape((-1,1)))
        count = np.zeros((imu_freq_net,dim))
        np.add.at(count,events_ti,mask.reshape((-1,1)))
        count[count==0] = 1e-6
        return input/count

    def get_displacement_measurement(self, net_gyr_w, net_acc_w, pe_ts, arch_type, ts_ev=None, R_gla_ev=None, R_0_ev=None,
                                    pos_0_ev=None, vel_0_ev=None, Rs_net_w=None,imu_freq_net=None, net_samples=None,
                                    event_based_input=False,base_event_stack=False,geodesic_event=False,
                                    integration_imu_frame=False, rot_component_weight = 2.0,
                                    se3_events=False, contrast_threshold=0.01,
                                    polarity_input=False,only_polarity_input=False, clip_small_disp=False,
                                    vio_preint = False, R_vio=None, p_vio=None, vel_vio=None):
        with torch.no_grad():
            if arch_type == 'rnin_vio_model_lstm':
                net_acc_w = net_acc_w - np.array([0.0,0.0,9.805]).reshape((1,-1))
            features = np.concatenate([net_gyr_w, net_acc_w], axis=1)  # N x 6

            ## for TLIO and non event models - interpolation 
            imu_freq_net = int(imu_freq_net)
            net_samples = int(net_samples)
            subsample = False
            if imu_freq_net!=net_samples and integration_imu_frame==False and event_based_input==False and base_event_stack==False and subsample==False:
                ## interpolating in world frame
                old_ts  = ts_ev
                ts = np.zeros((imu_freq_net,old_ts.shape[-1]))
                old_feat = features
                feats_new = np.zeros((imu_freq_net,old_feat.shape[1])) 
                ## directly interpolate
                ts = np.linspace(old_ts[1], old_ts[-1], imu_freq_net, endpoint=True).reshape((-1,1)) ## interpolating always to 200
                # print(old_ts[i,-1], ts[i,-1], imu_freq)
                assert old_ts[-1] == ts[-1]
                feats_new = interpolate.interp1d(old_ts.reshape((-1)), old_feat, axis=0)(ts.reshape((-1)))
                features = feats_new

            elif imu_freq_net!=net_samples and integration_imu_frame and event_based_input==False and base_event_stack==False:
                ## interpolating in imu frame
                old_ts  = ts_ev
                ts = np.zeros((imu_freq_net,old_ts.shape[-1]))
                old_feat = features
                feats_new = np.zeros((imu_freq_net,old_feat.shape[1])) 
                gyro = np.einsum('tji,tj->ti',R_gla_ev,old_feat[:,:3])
                accel = np.einsum('tji,tj->ti',R_gla_ev,old_feat[:,3:])

                ## interpolation
                ts = np.linspace(old_ts[0], old_ts[-1], imu_freq_net, endpoint=True).reshape((-1,1)) ## interpolating always to 200
                # # print(old_ts[i,-1], ts[i,-1], imu_freq)
                assert old_ts[-1] == ts[-1]
                feats_new[:,:3] = interpolate.interp1d(old_ts.reshape((-1)), gyro, axis=0)(ts.reshape((-1)))
                feats_new[:,3:] = interpolate.interp1d(old_ts.reshape((-1)), accel, axis=0)(ts.reshape((-1)))
                ## need to interpolate even the orientation
                vio_q_slerp = Slerp(old_ts.reshape((-1)), Rotation.from_matrix(R_gla_ev))
                targ_R_gla_ev = vio_q_slerp(ts.reshape((-1))).as_matrix()
                feats_new[:,:3] = np.einsum('tik,tk->ti',targ_R_gla_ev,feats_new[:,:3]) 
                feats_new[:,3:] = np.einsum('tik,tk->ti',targ_R_gla_ev,feats_new[:,3:]) 
                features = feats_new
            
            elif base_event_stack:#elif imu_freq_net!=net_samples and base_event_stack:
                features = self.gen_base_event_stack(ts_ev,features, imu_freq_net)


            elif event_based_input:
                if se3_events:
                    ts = ts_ev*1e-6
                    gyro = features[:,:3]
                    accel = features[:,3:]
                    gyro_i = np.einsum('tji,tj->ti', R_gla_ev, gyro)
                    accel_i = np.einsum('tji,tj->ti', R_gla_ev, accel) 


                    gyro = np.einsum('tij,tj->ti', Rs_net_w, gyro_i)
                    accel = np.einsum('tij,tj->ti', Rs_net_w, accel_i)

                    if polarity_input:
                        events, tk, Tk = imu_preintegration_and_se3_events(ts, np.expand_dims(R_0_ev,0), np.expand_dims(pos_0_ev[:,0],0), np.expand_dims(vel_0_ev[:,0],0), gyro_i, accel_i, gyro,accel, 
                                                            threshold=contrast_threshold, add_polarity=polarity_input,vio_preint = vio_preint, R_vio=R_vio, p_vio=p_vio, 
                                                            vel_vio=vel_vio)
                        
                        fl_events = np.zeros((2, 13))
                        if len(events) == 0:
                            ## calculate polarity
                            T_0 = np.eye(4)
                            T_0[:3,:3] = np.expand_dims(R_0_ev,0)[0].copy()
                            T_0[:3,3:4] = np.expand_dims(pos_0_ev[:,0],0)[0].copy().reshape((-1,1))
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

                        events.extend(fl_events)

                        if len(events) == 0:
                            events = np.zeros((1, 13))
                        else:
                            events = np.stack(events)
                            events = events[events[:, 0].argsort()]
                        features = []
                        features.append(self.generate_event_stack(events, ts,imu_freq_net, se3=True, start_idx=1))
                        
                        pol_ev = self.generate_event_stack(events, ts, imu_freq_net, se3=True,start_idx = 7)
                        features.append(pol_ev/(np.linalg.norm(pol_ev, axis=0).reshape((1,-1))+1e-4))
                        features = np.concatenate(features, axis=1)
                    else:
                        events, _, Tk = imu_preintegration_and_se3_events(ts, np.expand_dims(R_0_ev,0), np.expand_dims(pos_0_ev[:,0],0), np.expand_dims(vel_0_ev[:,0],0), gyro_i, accel_i, gyro,accel, 
                                                            threshold=contrast_threshold)
                        if len(events) == 0:
                            events = np.zeros((40, 7))
                        else:
                            events = np.stack(events)
                            events = events[events[:, 0].argsort()]

                        events = np.concatenate([
                            np.concatenate([ts[0].reshape((1,1)), np.array(features[0]).reshape((1,-1))], axis=-1).reshape((1,-1)),
                            events,
                            np.concatenate([ts[-1].reshape((1,1)), np.array(features[-1]).reshape((1,-1))], axis=-1).reshape((1,-1))
                        ])
                        
                       
                        features = self.generate_event_stack(events, ts, imu_freq_net, se3=True, start_idx = 1)

                else:
                    ts = ts_ev*1e-6
                    gyro = features[:,:3]
                    accel = features[:,3:]
                    gyro_i = np.einsum('tji,tj->ti', R_gla_ev, gyro)
                    accel_i = np.einsum('tji,tj->ti', R_gla_ev, accel) 

                    r_random_ball = random_sample_ball(radius=0.01)[0]
                    p_random_ball = random_sample_ball(radius=0.01)[0]

                    gyro = np.einsum('tij,tj->ti', Rs_net_w, gyro_i)
                    accel = np.einsum('tij,tj->ti', Rs_net_w, accel_i)

                    events = imu_preintegration_and_geodesic_events(ts, np.expand_dims(R_0_ev,0), np.expand_dims(pos_0_ev[:,0],0), np.expand_dims(vel_0_ev[:,0],0), gyro_i, accel_i, gyro,accel, threshold=0.01,
                                                                    rot_component_weight=rot_component_weight, use_rotation=geodesic_event, init_ref_random=False,
                                                                    p_random_ball=p_random_ball, r_random_ball=r_random_ball)
                    if len(events) == 0:
                        events = np.zeros((0, 10))
                    else:
                        events = np.stack(events)
                        events = events[events[:, 0].argsort()]
                        events = events[:, :-1]

                    
                    events = np.concatenate([
                        np.concatenate([np.array([ts[0], 0, 0 , 1]).reshape((1,-1)), np.array(features[0]).reshape((1,-1))], axis=-1).reshape((1,-1)),
                        events,
                        np.concatenate([np.array([ts[-1], 0, 0 , 1]).reshape((1,-1)), np.array(features[-1]).reshape((1,-1))], axis=-1).reshape((1,-1))
                    ])
                        
 
                    features = self.generate_event_stack(events, ts, imu_freq_net,new=True)

                ## finding yaw to put it in gravity aligned
                ri_z = compute_euler_from_matrix(R_0_ev, "xyz", extrinsic=True)[0, 2]
                Ri_z = np.array(
                    [
                        [np.cos(ri_z), -(np.sin(ri_z)), 0],
                        [np.sin(ri_z), np.cos(ri_z), 0],
                        [0, 0, 1],
                    ]
                )

                features[:,:3] = np.einsum('ji,tj->ti', Ri_z,features[:,:3])
                features[:,3:6] = np.einsum('ji,tj->ti', Ri_z, features[:,3:6])
                if polarity_input:
                    features[:,6:9] = np.einsum('ji,tj->ti', Ri_z,features[:,6:9])
                    features[:,9:12] = np.einsum('ji,tj->ti', Ri_z, features[:,9:12])



            net_gyr_w = features[:,:3]
            net_acc_w = features[:,3:6]
            if only_polarity_input:
                features = features[:, 6:]
            features_t = torch.unsqueeze(
                torch.from_numpy(features.T).float().to(self.device), 0
            )  # 1 x 6 x N
            pe_ts = torch.unsqueeze(torch.from_numpy(pe_ts).float().to(self.device), 0)
           
            # print('features shape:',features.shape)
            # print('features_t shape:', features_t.shape)

            
            if arch_type == 'rnin_vio_model_lstm':
                features_t = features_t.unsqueeze(dim=1)
                ## rnin is gravity compensated
                netargs = [features_t]
                pred, pred_cov = self.net(*netargs)
                outputs = (pred.squeeze(dim=1), pred_cov.squeeze(dim=1))
            else:
                netargs = [features_t]
                outputs = self.net(*netargs)


            if type(outputs) == tuple:  # Legacy
                meas, meas_cov = outputs
            elif type(outputs) == dict:  # New output format
                meas, meas_cov = outputs["pred"], outputs["pred_log_std"]
                # If this is the case, the network predicts over the whole window at high frequency.
                # TODO utilize the whole window measurements. May improve.
                if meas.dim() == 3:
                    meas = meas[:, -1]
                    meas_cov = meas_cov[:, -1]
                    
            
        

            assert meas.dim() == 2  # [B,3]
            assert meas_cov.dim() == 2

            meas = meas.cpu().detach().numpy()
            meas_cov[meas_cov < -4] = -4  # exp(-3) =~ 0.05
            meas_cov = DiagonalParam.vec2Cov(meas_cov).cpu().detach().numpy()[0, :, :]
            meas = meas.reshape((3, 1))

            # Our equivalent of zero position update (TODO need stronger prior to keep it still)
            if clip_small_disp and np.linalg.norm(meas) < 0.001:
                meas = 0 * meas
                # meas_cov = 1e-6 * np.eye(3)

            return meas, meas_cov
