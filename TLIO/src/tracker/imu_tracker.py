#!/usr/bin/env python3

import json
from typing import Optional

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from tracker.imu_buffer import ImuBuffer
from tracker.imu_calib import ImuCalib
from tracker.meas_source_torchscript import MeasSourceTorchScript
from tracker.scekf import ImuMSCKF
from utils.dotdict import dotdict
from utils.from_scipy import compute_euler_from_matrix
from utils.logging import logging
from utils.math_utils import mat_exp
from scipy.special import erfinv
from utils.math_utils import Jr_exp, hat, mat_exp, mat_exp_vec, mat_log, rot_2vec
from scipy import interpolate
from scipy.spatial.transform import Slerp

def geodesic_distance(R0, R1, t0, t1, b):
    dt = t1 - t0
    dr = mat_log(R0 @ R1.T)
    d = np.sqrt((dt**2).sum() + b * (dr**2).sum())
    return d

def geodesic(w0, w1, p0, p1, b):
    return np.sqrt(b * np.linalg.norm(w0 - w1) ** 2 + np.linalg.norm(p0 - p1) ** 2)

@jit(nopython=True)
def geodesic_events_translation(p_ref, p0, p1, t0, t1, threshold):
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
        bil = (p_ref - p0).dot(n_p01) / (np.linalg.norm(p_ref - p0)+1e-10)

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)

        # gather event
        event = np.zeros((4,))
        event[0] = t_i
        event[1:4] = polarity
        events.append(event)

    return p_ref, events


@jit(nopython=True)
def geodesic_events_rotation(R_ref, R0, R1, t0, t1, threshold):
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
        bil = mat_log(R0.T @ R_ref).dot(n_w01) / (np.linalg.norm(mat_log(R0.T @ R_ref))+1e-10)

        # get timestamp and measurements for timestamp
        t_i = t0 + bil * (t1 - t0)

        # gather event
        event = np.zeros((4,))
        event[0] = t_i
        event[1:4] = polarity
        events.append(event)

    return R_ref, events


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


@jit(nopython=True)
def imu_preintegration_and_geodesic_events(t, R, pos, threshold=0.01, rot_component_weight=2, use_rotation=True,
                                           init_ref_random=False, r_random_ball=None, p_random_ball=None):
    events = []

    Rk = R[0].copy()
    pk = pos[0].copy()

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
        Rd = R[i] 
        pd = pos[i]

        # generate events for translation
        p_ref, translation_events = geodesic_events_translation(p_ref, pk, pd, tk, t[i], threshold)
        events.extend(translation_events)

        if use_rotation:
            # generate events for rotation
            R_ref, rotation_events = geodesic_events_rotation(R_ref, Rk, Rd, tk, t[i], rot_component_weight * threshold)
            events.extend(rotation_events)

        Rk = Rd
        pk = pd
        tk = t[i]

    return events

def generate_event_stack(events, base_ts, new=False):
        if new==True:
            events_meas = events[:, 4:].copy()
            ts = events[:, 0].copy()
        else:
            ts = np.array([event[0] for event in events]).reshape((-1))
            p = np.array([event[1] for event in events]).reshape((-1))
            events_meas = np.array([event[2] for event in events]).reshape((-1,6))

        ### event stack grid generation
        B = 200 # number of bins to discretize the volume grid
        events_ti = np.linspace(0, B-1, ts.shape[0], endpoint=True).astype(int)

        input = np.zeros((200,6))
        np.add.at(input,events_ti,events_meas)
        count = np.zeros((200,6))
        np.add.at(count,events_ti,int(1))
        count[count==0] = 1e-6
        return input/count

class ImuTracker:
    """
    ImuTracker is responsible for feeding the EKF with the correct data
    It receives the imu measurement, fills the buffer, runs the network with imu data in buffer
    and drives the filter.
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        net_samples,
        imu_freq_net,
        update_freq,
        filter_tuning_cfg,
        imu_calib: Optional[ImuCalib] = None,
        initial_velocity_debug=False,
        event_based_input = False,
        base_event_stack = False,
        geodesic_event = False,
        integration_imu_frame = False,
        rot_component_weight = 2.0,
        se3_events=False, 
        contrast_threshold=0.01, 
        extra = 0,
        polarity_input = False,
        only_polarity_input = False,
        vio_initialise_preint = False,
        vio_preint = False,
        force_cpu=False,
    ):
        self.initial_velocity_debug = initial_velocity_debug
        self.event_based_input = event_based_input
        self.base_event_stack = base_event_stack
        self.geodesic_event = geodesic_event
        self.integration_imu_frame = integration_imu_frame
        self.net_samples = net_samples
        self.rot_component_weight = rot_component_weight
        self.se3_events = se3_events
        self.contrast_threshold = contrast_threshold
        self.polarity_input = polarity_input
        self.only_polarity_input = only_polarity_input
        self.extra = extra
        self.vio_initialise_preint = vio_initialise_preint
        self.vio_preint = vio_preint

        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = imu_freq_net
            config_from_network["past_time"] = data_json["past_time"]
            config_from_network["window_time"] = data_json["window_time"]
            config_from_network["arch"] = data_json["arch"]
            config_from_network["input_dim"] = data_json["input_dim"]
            config_from_network["output_dim"] = data_json["output_dim"]
        self.arch = config_from_network["arch"]
        self.log_data = True
        # frequencies and sizes conversion
        if not (
            config_from_network.past_time * config_from_network.imu_freq_net
        ).is_integer():
            raise ValueError(
                "past_time cannot be represented by integer number of IMU data."
            )
        if not (
            config_from_network.window_time * config_from_network.imu_freq_net
        ).is_integer():
            raise ValueError(
                "window_time cannot be represented by integer number of IMU data."
            )
        self.imu_freq_net = (
            config_from_network.imu_freq_net
        )  # imu frequency as input to the network
        self.past_data_size = int(
            config_from_network.past_time * config_from_network.imu_freq_net
        )
        self.disp_window_size = int(
            config_from_network.window_time * config_from_network.imu_freq_net
        )
        self.net_input_size = self.disp_window_size + self.past_data_size

        # EXAMPLE :
        # if using 200 samples with step size 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        # if using 400 samples with 200 past data and clone_every_n_netimu_sample 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        if not (config_from_network.imu_freq_net / update_freq).is_integer():
            raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError(
                "window_time cannot be represented by integer number of updates."
            )
        self.update_freq = update_freq
        self.clone_every_n_netimu_sample = int(
            config_from_network.imu_freq_net / update_freq
        )  # network inference/filter update interval
        assert (
            config_from_network.imu_freq_net % update_freq == 0
        )  # imu frequency must be a multiple of update frequency
        self.update_distance_num_clone = int(
            config_from_network.window_time * update_freq
        )

        # time
        self.dt_interp_us = int(1.0 / self.net_samples * 1e6)
        self.dt_update_us = int(
            1.0 / self.update_freq * 1e6
        )  # multiple of interpolation interval

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.past_time + config_from_network.window_time} = {config_from_network.past_time} + {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} = {self.past_data_size} + {self.disp_window_size} (samples)"
        )
        logging.info("IMU interpolation frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            "Filter update stride state number: %i" % self.update_distance_num_clone
        )
        logging.info(
            f"Interpolating IMU measurement every {self.dt_interp_us}us for the network input"
        )

        # IMU initial calibration
        self.icalib = imu_calib
        self.filter_tuning_cfg = filter_tuning_cfg # Config
        # MSCKF
        self.filter = ImuMSCKF(filter_tuning_cfg)

        net_config = {"in_dim": (self.past_data_size + self.disp_window_size) // 32 + 1, "input_dim":config_from_network["input_dim"],
                      "output_dim":config_from_network["output_dim"]}
        self.meas_source = MeasSourceTorchScript(model_path,  config_from_network["arch"], net_config, force_cpu)

        self.imu_buffer = ImuBuffer(self.extra)

        #  This callback is called at first update if set
        self.callback_first_update = None
        # This callback can be use to bypass network use for measurement
        self.debug_callback_get_meas = None

        # keep track of past timestamp and measurement
        self.last_t_us = -1

        # keep track of the last measurement received before next interpolation time
        self.t_us_before_next_interpolation = -1
        self.last_acc_before_next_interp_time = None
        self.last_gyr_before_next_interp_time = None

        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.has_done_first_update = False

        self.yaw_aligned = True #local yaw aligned frame
        self.local_frame = False #local frame with full R (not only yaw)
        # self.P = np.array([[0,1,0],[1,0,0],[0,0,1]])
        # self.testing_reflections = False

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

    @jit(forceobj=True, parallel=False, cache=False)
    def _get_imu_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        # extract corresponding network input data
        net_tus_begin = t_begin_us
        net_tus_end = t_end_us - self.dt_interp_us
        step = 0
        if self.imu_freq_net!=self.net_samples:
            step=1
        net_acc, net_gyr, net_tus = self.imu_buffer.get_data_from_to(
            net_tus_begin, net_tus_end, step
        )
        ts_ind = np.arange(1, net_tus.shape[0]+1)
        pe_ts = self.positional_encoding(ts_ind.reshape((-1,1)), num_frequencies=4, incl_input=True)

        # assert net_gyr.shape[0] == self.net_input_size
        # assert net_acc.shape[0] == self.net_input_size
        # get data from filter
        R_0, pos_0, vel_0, vel_0_ps = self.filter.get_past_state(t_oldest_state_us)  # 3 x 3


        ## frame change for network input
        Ri_z = np.eye(3) ## if we want the network input to be in world frame
        if self.yaw_aligned: ## changing network input to loacl yaw aligned frame
            # change the input of the network to be in local frame
            # print('Entered local gravity aligned frame neural imu samples')
            ri_z = compute_euler_from_matrix(R_0, "xyz", extrinsic=True)[
                0, 2
            ]
            Ri_z = np.array(
                [
                    [np.cos(ri_z), -(np.sin(ri_z)), 0],
                    [np.sin(ri_z), np.cos(ri_z), 0],
                    [0, 0, 1],
                ]
            )

        if self.local_frame:
             # change the input of the network to be in local frame
            # print('Entered local frame neural imu samples')
            Ri_z = R_0 ## taking the first timestep

        
        R_oldest_state_wfb = Ri_z.T @ R_0

        bg = self.filter.state.s_bg
        # dynamic rotation integration using filter states
        # Rs_net will contains delta rotation since t_begin_us
        Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))  # N x 3 x 3
        Rs_bofbi[0, :, :] = np.eye(3)
        for j in range(1, net_tus.shape[0]):
            dt_us = net_tus[j] - net_tus[j - 1]
            dR = mat_exp((net_gyr[j, :].reshape((3, 1)) - bg) * dt_us * 1e-6)
            Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

        # find delta rotation index at time ts_oldest_state
        oldest_state_idx_in_net = np.where(net_tus == t_oldest_state_us)[0][0]

        # rotate all Rs_net so that (R_oldest_state_wfb @ (Rs_bofbi[idx].inv() @ Rs_bofbi[i])
        # so that Rs_net[idx] = R_oldest_state_wfb
        R_bofboldstate = (
            R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        )  # [3 x 3]
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
        net_acc_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_acc)  # N x 3
        net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)  # N  x 3

        Rs_net_w = (
            R_0 @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        )  # [3 x 3]
        Rs_net_w = np.einsum("ip,tpj->tij", Rs_net_w, Rs_bofbi)


        return net_gyr_w, net_acc_w, pe_ts, net_tus, Rs_net_wfb, R_0, pos_0, vel_0, Rs_net_w, vel_0_ps

    def _compensate_measurement_with_initial_calibration(self, gyr_raw, acc_raw):
        if self.icalib:
            #logging.info("Using bias from initial calibration")
            init_ba = self.icalib.accelBias
            init_bg = self.icalib.gyroBias
            # calibrate raw imu data
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )  # removed offline bias and scaled
        else:
            #logging.info("Using zero bias")
            init_ba = np.zeros((3, 1))
            init_bg = np.zeros((3, 1))
            acc_biascpst, gyr_biascpst = acc_raw, gyr_raw
        return gyr_biascpst, acc_biascpst, init_bg, init_ba

    def _after_filter_init_member_setup(self, t_us, gyr_biascpst, acc_biascpst):
        self.next_interp_t_us = t_us
        self.next_aug_t_us = t_us
        self._add_interpolated_imu_to_buffer(acc_biascpst, gyr_biascpst, t_us)
        self.next_aug_t_us = t_us + self.dt_update_us

        self.last_t_us = t_us

        self.t_us_before_next_interpolation = t_us
        self.last_acc_before_next_interp_time = acc_biascpst
        self.last_gyr_before_next_interp_time = gyr_biascpst

    def init_with_state_at_time(self, t_us, R, v, p, gyr_raw, acc_raw):
        assert R.shape == (3, 3)
        assert v.shape == (3, 1)
        assert p.shape == (3, 1)

        logging.info(f"Initializing filter at time {t_us*1e-6}")
        (
            gyr_biascpst,
            acc_biascpst,
            init_bg,
            init_ba,
        ) = self._compensate_measurement_with_initial_calibration(gyr_raw, acc_raw)
        self.filter.initialize_with_state(t_us, R, v, p, init_ba, init_bg)
        self._after_filter_init_member_setup(t_us, gyr_biascpst, acc_biascpst)
        return False

    def _init_without_state_at_time(self, t_us, gyr_raw, acc_raw):
        assert isinstance(t_us, int)
        logging.info(f"Initializing filter at time {t_us*1e-6}")
        (
            gyr_biascpst,
            acc_biascpst,
            init_bg,
            init_ba,
        ) = self._compensate_measurement_with_initial_calibration(gyr_raw, acc_raw)
        self.filter.initialize(t_us, acc_biascpst, init_ba, init_bg)
        self._after_filter_init_member_setup(t_us, gyr_biascpst, acc_biascpst)

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw):
        assert isinstance(t_us, int)
        # print(t_us - self.last_t_us)
        # if t_us - self.last_t_us > 3e3: ## disabling for 100Hz seq
            
            # logging.warning(f"Big IMU gap : {t_us - self.last_t_us}us")

        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw)
        else:
            self._init_without_state_at_time(t_us, gyr_raw, acc_raw)
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw):
        """
        For new IMU measurement, after the filter has been initialized
        """
        assert isinstance(t_us, int)

        # Eventually calibrate
        if self.icalib:
            # calibrate raw imu data with offline calibation
            # this is used for network feeding
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )  # removed offline bias and scaled

            # calibrate raw imu data with offline calibation scale
            # this is used for the filter. By not applying offline bias
            # we expect the filter to estimate bias similar to the offline
            # calibrated one
            acc_raw, gyr_raw = self.icalib.scale_raw(
                acc_raw, gyr_raw
            )  # only offline scaled - into the filter
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # decide if we need to interpolate imu data or do update
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_augmentation_and_update = t_us >= self.next_aug_t_us

        # if augmenting the state, check that we compute interpolated measurement also
        assert (
            do_augmentation_and_update and do_interpolation_of_imu
        ) or not do_augmentation_and_update, (
            "Augmentation and interpolation does not match!"
        )

        # augmentation propagation / propagation
        # propagate at IMU input rate, augmentation propagation depends on t_augmentation_us
        t_augmentation_us = self.next_aug_t_us if do_augmentation_and_update else None

        # IMU interpolation and data saving for network (using compensated IMU)
        if do_interpolation_of_imu:
            self._add_interpolated_imu_to_buffer(acc_biascpst, gyr_biascpst, t_us)

        self.filter.propagate(
            acc_raw, gyr_raw, t_us, t_augmentation_us=t_augmentation_us
        )
        # filter update
        did_update = False
        if do_augmentation_and_update:
            did_update = self._process_update(t_us)
            # plan next update/augmentation of state
            self.next_aug_t_us += self.dt_update_us

        # set last value memory to the current one
        self.last_t_us = t_us

        if t_us < self.t_us_before_next_interpolation:
            self.t_us_before_next_interpolation = t_us
            self.last_acc_before_next_interp_time = acc_biascpst
            self.last_gyr_before_next_interp_time = gyr_biascpst

        return did_update

    def _process_update(self, t_us):
        logging.debug(f"Upd. @ {t_us * 1e-6} | Ns: {self.filter.state.N} ")
        # get update interval t_begin_us and t_end_us
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1
        ]
        if t_oldest_state_us == 1248952369:
            print(t_oldest_state_us)
        t_begin_us = t_oldest_state_us - self.dt_interp_us * self.past_data_size
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        # If we do not have enough IMU data yet, just wait for next time
        if  t_end_us*1e-6 == 622.482485:#t_end_us*1e-6 > 763.5 or
            print(t_oldest_state_us)
        if (t_begin_us < self.imu_buffer.net_t_us[self.extra]) or (t_end_us>self.imu_buffer.net_t_us[-1]):
            return False
        # print(t_end_us,self.imu_buffer.net_t_us[-5:])
        # initialize with vio at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us
        if self.debug_callback_get_meas!=None and self.initial_velocity_debug==False and self.vio_initialise_preint==False and self.vio_preint==False:
            meas, meas_cov = self.debug_callback_get_meas(t_oldest_state_us, t_end_us)
        else:  # using network for measurements
            net_gyr_w, net_acc_w, pe_ts,ts_ev, R_net_gla,R_0_ev, pos_0_ev, vel_0_ev, Rs_net_w,vel_0_ps = self._get_imu_samples_for_network(
                t_begin_us, t_oldest_state_us, t_end_us
            )
            if self.initial_velocity_debug:
                _, _, vel_0_ev = self.debug_callback_get_meas(t_oldest_state_us, t_end_us, init_vel=True)
            R_vio, p_vio, vel_vio = None, None, None
            if self.vio_initialise_preint or self.vio_preint:
                net_tus_begin = t_begin_us
                net_tus_end = t_end_us - self.dt_interp_us
                step = 0
                if self.imu_freq_net!=self.net_samples:
                    step=1
                acc, gyro, net_tus = self.imu_buffer.get_data_from_to(
                    net_tus_begin, net_tus_end, step
                )
                R_vio, p_vio, vel_vio = self.debug_callback_get_meas(t_oldest_state_us, t_end_us, init_vel=False, vio_preint=True, net_tus = net_tus)
                net_gyr_w, net_acc_w, ts_ev, R_net_gla,R_0_ev, pos_0_ev, vel_0_ev, Rs_net_w = None, None,None, None,None, None, None, None
                ts_ev = net_tus
                R_0_ev = R_vio[0]
                pos_0_ev = p_vio[0].reshape((-1,1))
                vel_0_ev = vel_vio[0]
                ## world frame
                ri_z = compute_euler_from_matrix(R_0_ev, "xyz", extrinsic=True)[0, 2]
                Ri_z = np.array(
                    [
                        [np.cos(ri_z), -(np.sin(ri_z)), 0],
                        [np.sin(ri_z), np.cos(ri_z), 0],
                        [0, 0, 1],
                    ]
                )

                ## gravity alignment
                R_net_gla = np.einsum("ip,tpj->tij", Ri_z.T, R_vio)
                net_acc_w = np.einsum("tij,tj->ti", R_net_gla, acc)  # N x 3
                net_gyr_w = np.einsum("tij,tj->ti", R_net_gla, gyro)  # N  x 3

                ## yaw angle
                Rs_net_w = R_vio.copy()
                
            vel_0_ev = vel_0_ps
            if self.vio_initialise_preint: ## this is only for debugging
                # R_0_ev = R_vio[0]
                pos_0_ev = p_vio[0].reshape((-1,1))
                vel_0_ev = vel_vio[0]
            meas, meas_cov = self.meas_source.get_displacement_measurement(
                net_gyr_w, net_acc_w, pe_ts, self.arch, ts_ev, R_net_gla,R_0_ev, pos_0_ev, vel_0_ev.reshape((-1,1)), Rs_net_w,
                self.imu_freq_net, self.net_samples,self.event_based_input,self.base_event_stack,self.geodesic_event,
                self.integration_imu_frame, rot_component_weight = self.rot_component_weight, 
                se3_events=self.se3_events, contrast_threshold=self.contrast_threshold,
                polarity_input = self.polarity_input, only_polarity_input = self.only_polarity_input,
                vio_preint = self.vio_preint, R_vio=R_vio, p_vio=p_vio, vel_vio=vel_vio
            )
        # filter update
        self.filter.update(meas, meas_cov, t_oldest_state_us, t_end_us)
        self.has_done_first_update = True
        # marginalization of all past state with timestamp before or equal ts_oldest_state
        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        cut_idx = oldest_idx
        logging.debug(f"marginalize {cut_idx}")
        self.filter.marginalize(cut_idx)
        self.imu_buffer.throw_data_before(t_begin_us)
        return True

    def _add_interpolated_imu_to_buffer(self, acc_biascpst, gyr_biascpst, t_us):
        while t_us>=self.next_interp_t_us:
            self.imu_buffer.add_data_interpolated(
                self.t_us_before_next_interpolation,
                t_us,
                self.last_gyr_before_next_interp_time,
                gyr_biascpst,
                self.last_acc_before_next_interp_time,
                acc_biascpst,
                self.next_interp_t_us,
            )
            self.next_interp_t_us += self.dt_interp_us
