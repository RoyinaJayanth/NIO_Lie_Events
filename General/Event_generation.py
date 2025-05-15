import numpy as np
from utils_tlio.math_utils import mat_exp, mat_log, exp_SE3 as mat_exp_se3, log_SE3 as mat_log_se3 # type: ignore

from numba import jit
from scipy.special import erfinv


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