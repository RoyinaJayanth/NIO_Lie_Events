from os import path as osp
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from utils.logging import logging
from utils.math_utils import unwrap_rpy, wrap_rpy, inv_SE3


class DataIO:
    def __init__(self):
        # raw dataset - ts in us
        self.ts_all = None
        self.gyr_all = None
        self.acc_all = None
        self.dataset_size = None
        self.init_ts = None
        self.R_init = np.eye(3)
        # vio data
        self.vio_ts = None
        self.vio_p = None
        self.vio_v = None
        self.vio_eul = None
        self.vio_R = None
        self.vio_rq = None
        self.vio_ba = None
        self.vio_bg = None
        # attitude filter data
        self.filter_ts = None
        self.filter_eul = None

    def load_all(self, dataset, args):
        """
        load timestamps, accel and gyro data from dataset
        """
        imu_data = pd.read_csv(osp.join(args.root_dir, dataset, "imu_samples_0.csv"))
        ts_all = np.copy(imu_data.iloc[:,0]) * 1e-3
        gyr_all = np.copy(imu_data.iloc[:,2:5])
        acc_all = np.copy(imu_data.iloc[:,5:8])
        if args.start_from_ts is not None:
            idx_start = np.where(ts_all >= args.start_from_ts)[0][0]
        else:
            idx_start = 50
        self.ts_all = ts_all[idx_start:]
        self.gyr_all = gyr_all[idx_start:, :]
        self.acc_all = acc_all[idx_start:, :]
        self.dataset_size = self.ts_all.shape[0]
        self.init_ts = self.ts_all[0]

    def load_vio(self, dataset, args):
        """
        load ts, p, q, v from vio states, load ba and bg from calibration states
        """
        logging.info(
            "loading vio states from "
            + osp.join(args.root_dir, dataset, "imu0_resampled.npy")
        )
        data = np.load(osp.join(args.root_dir, dataset, "imu0_resampled.npy"))
        self.vio_ts_us = data[:, 0]
        self.vio_ts = data[:, 0] * 1e-6
        self.vio_rq = data[:,-10:-6]
        self.vio_p = data[:,-6:-3]
        self.vio_v = data[:,-3:]
        vio_r = Rotation.from_quat(self.vio_rq)
        self.vio_R = vio_r.as_matrix()
        self.vio_eul = vio_r.as_euler("xyz", degrees=True)
        imu_ts = self.ts_all * 1e-6
        idx = np.max(np.where(imu_ts<=self.vio_ts[-1])[0])
        # self.ts_all = self.ts_all[:idx]
        self.dataset_size = self.ts_all[:idx].shape[0]




    def get_datai(self, idx):
        ts = self.ts_all[idx] * 1e-6  # s
        acc = self.acc_all[idx, :].reshape((3, 1))
        gyr = self.gyr_all[idx, :].reshape((3, 1))
        return ts, acc, gyr

    def get_meas_from_vio(self, ts_oldest_state, ts_end, init_vel=False, vio_preint=False, net_tus = None):
        """
        helper function This extracts a fake measurement from vio,
        can be used for debug to bypass the network
        """
        # obtain vio_Ri for rotating to relative frame
        ts_oldest_state = ts_oldest_state*1e-6
        ts_end = ts_end*1e-6
        idx_left = np.where(self.vio_ts < ts_oldest_state)[0][-1]
        
        if vio_preint:
            net_tus = net_tus*1e-6
            idx_right = np.where(self.vio_ts > ts_end)[0][0]
            interp_vio_ts = self.vio_ts[idx_left : idx_right + 1]
            interp_vio_eul = self.vio_eul[idx_left : idx_right + 1, :]
            vio_euls_uw = unwrap_rpy(interp_vio_eul)
            vio_eul_uw = interp1d(interp_vio_ts, vio_euls_uw, axis=0)(net_tus)
            vio_eul = np.deg2rad(wrap_rpy(vio_eul_uw))
            R = Rotation.from_euler("xyz", vio_eul, degrees=True).as_matrix()
            vio_interp = interp1d(self.vio_ts, self.vio_p, axis=0)(net_tus)
            # express in gravity aligned frame by normalizing on yaw
            vio_interp_vel = interp1d(self.vio_ts, self.vio_v, axis=0)(net_tus)
            return R, vio_interp, vio_interp_vel
        
        idx_right = np.where(self.vio_ts > ts_oldest_state)[0][0]
        interp_vio_ts = self.vio_ts[idx_left : idx_right + 1]
        interp_vio_eul = self.vio_eul[idx_left : idx_right + 1, :]
        vio_euls_uw = unwrap_rpy(interp_vio_eul)
        vio_eul_uw = interp1d(interp_vio_ts, vio_euls_uw, axis=0)(ts_oldest_state)
        vio_eul = np.deg2rad(wrap_rpy(vio_eul_uw))
        ts_interp = np.array([ts_oldest_state, ts_end])
        vio_interp = interp1d(self.vio_ts, self.vio_p, axis=0)(ts_interp)
        vio_meas = vio_interp[1] - vio_interp[0]  # simulated displacement measurement
        meas_cov = np.diag(np.array([1e-2, 1e-2, 1e-2]))  # [3x3]
        # express in gravity aligned frame by normalizing on yaw
        Ri_z = Rotation.from_euler("z", vio_eul[2]).as_matrix()
        meas = Ri_z.T.dot(vio_meas.reshape((3, 1)))
        if init_vel:
            vio_interp_vel = interp1d(self.vio_ts, self.vio_v, axis=0)(ts_interp)
            meas_vel = vio_interp_vel[0]  # simulated velocity measurement
            # meas_vel = Ri_z.T.dot(meas_vel.reshape((3, 1))) ## don't do gravity
            return meas, meas_cov, meas_vel
        return meas, meas_cov
