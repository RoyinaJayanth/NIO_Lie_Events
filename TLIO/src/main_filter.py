"""
TLIO Stochastic Cloning Extended Kalman Filter
Input: IMU data
Measurement: window displacement estimates from networks
Filter states: position, velocity, rotation, IMU biases
"""

import argparse
import datetime
import json
import os

# silence NumbaPerformanceWarning
import warnings
from pprint import pprint

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from tracker.imu_tracker_runner import ImuTrackerRunner
from utils.argparse_utils import add_bool_arg
from utils.logging import logging

import random
import torch

def seed_everything(seed=81):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser()

    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")

    io_groups.add_argument(
        "--root_dir", type=str, 
        default="../TLIO-master/local_data/tlio_golden", help="Path to data directory"
    )
    io_groups.add_argument("--dataset_number", type=int, default=None)
    io_groups.add_argument("--model_path", type=str, default="../TLIO-master/output/tlio_ev_se3p/checkpoint_best.pt")
    io_groups.add_argument("--model_param_path", type=str, default="../TLIO-master/output/tlio_ev_se3p/parameters.json") #, required=True
    io_groups.add_argument("--out_dir", type=str, default="../TLIO-master/output/tlio_ev_se3p/ekf_output")
    io_groups.add_argument("--out_filename", type=str, default="not_vio_state.txt")
    io_groups.add_argument("--save_as_npy", action="store_true")
    io_groups.add_argument("--sim_data_path", type=str, default="imu-sim.txt")
    io_groups.add_argument(
        "--start_from_ts", type=int, default=None
    )  # dataloader loading data from timestamp (us)
    io_groups.add_argument("--imu_freq_net", type=float, default="200.0")
    add_bool_arg(io_groups, "visualize", default=False, 
            help="Opens up a visualization window")
    add_bool_arg(io_groups, "erase_old_log", default=False)

    ## base and propagation IMU frequency - we only skip readings
    io_groups.add_argument("--imu_base_freq", type=float, default="1000.0")
    io_groups.add_argument("--propagation_freq", type=float, default="1000.0")
    io_groups.add_argument("--net_samples", type=float, default="200.0") ## changing frequencies

    # ----------------------- network params -----------------------
    net_groups = parser.add_argument_group("network")
    net_groups.add_argument("--cpu", action="store_true")
    add_bool_arg(net_groups, "initial_velocity_debug", default=False) ## this is to debug event generation
    add_bool_arg(net_groups, "event_based_input", default=True)
    add_bool_arg(net_groups, "base_event_stack", default=False)
    add_bool_arg(net_groups, "geodesic_event", default=False)
    net_groups.add_argument("--rot_component_weight", type=int, default="1") ## changing weights
    add_bool_arg(net_groups, "integration_imu_frame", default=False)

    # extra_imu_buffer_length - this is 1 when NN freq is same as ekf because we need one extra propagation step to do measurement update
    net_groups.add_argument("--extra_imu_buffer_length", type=int, default="0")## this is 1 for aria datasets
    add_bool_arg(net_groups, "polarity_input", default=True)# polarity_input
    add_bool_arg(net_groups, "only_polarity_input", default=False)# only_polarity_input
    # ----------------------- filter params -----------------------
    filter_group = parser.add_argument_group("filter tuning:")

    filter_group.add_argument("--update_freq", type=float, default=20.0)  # (Hz)

    ## contrast threshold for event generation
    filter_group.add_argument("--contrast_threshold", type=float, default="0.01") # 0.01 for 100 events and 0.005 for 200 events
    

    add_bool_arg(filter_group,"se3_events", default=True)

    ## debug preintegration during event generation
    add_bool_arg(filter_group,"vio_initialise_preint", default=False)
    add_bool_arg(filter_group,"vio_preint", default=False)

    filter_group.add_argument(
        "--sigma_na", type=float, default=np.sqrt(1e-3)
    )  # accel noise  m/s^2
    filter_group.add_argument(
        "--sigma_ng", type=float, default=np.sqrt(1e-4)
    )  # gyro noise  rad/s
    filter_group.add_argument(
        "--ita_ba", type=float, default=1e-4
    )  # accel bias noise  m/s^2/sqrt(s)
    filter_group.add_argument(
        "--ita_bg", type=float, default=1e-6
    )  # gyro bias noise  rad/s/sqrt(s)

    filter_group.add_argument(
        "--init_attitude_sigma", type=float, default=1.0 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument(
        "--init_yaw_sigma", type=float, default=0.1 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument("--init_vel_sigma", type=float, default=1.0)  # m/s
    filter_group.add_argument("--init_pos_sigma", type=float, default=0.001)  # m
    filter_group.add_argument(
        "--init_bg_sigma", type=float, default=0.0001
    )  # rad/s  0.001
    filter_group.add_argument("--init_ba_sigma", type=float, default=0.02)  # m/s^2  0.02
    filter_group.add_argument("--g_norm", type=float, default=9.81)

    filter_group.add_argument("--meascov_scale", type=float, default=10.0) #10.0

    add_bool_arg(
        filter_group, "initialize_with_vio", default=True
    )  # initialize state with gt state
    add_bool_arg(
        filter_group, "initialize_with_offline_calib", default=True
    )  # initialize bias state with offline calib or 0

    filter_group.add_argument(
        "--mahalanobis_fail_scale", type=float, default=0
    )  # if nonzero then mahalanobis gating test would scale the covariance by this scale if failed

    # ----------------------- debug params -----------------------
    debug_groups = parser.add_argument_group("debug")
    # covariance alternatives (note: if use_vio_meas is true, meas constant with default value 1e-4)
    add_bool_arg(debug_groups, "use_const_cov", default=False)
    debug_groups.add_argument(
        "--const_cov_val_x", type=float, default=np.power(0.1, 2.0)#0.1, 2.0
    )
    debug_groups.add_argument(
        "--const_cov_val_y", type=float, default=np.power(0.1, 2.0)#0.1, 2.0
    )
    debug_groups.add_argument(
        "--const_cov_val_z", type=float, default=np.power(0.1, 2.0)#0.1, 2.0
    )

    # measurement alternatives (note: if use_vio_meas is false, add_sim_meas_noise msust be false)
    add_bool_arg(
        debug_groups,
        "use_vio_meas",
        default=False,
        help='If using "vio" measurement for filter update instead of ouptut network',
    )
    add_bool_arg(debug_groups, "debug_using_vio_ba", default=False)
    add_bool_arg(
        debug_groups, "add_sim_meas_noise", default=False
    )  # adding noise on displacement measurement when using vio measurement
    debug_groups.add_argument(
        "--sim_meas_cov_val", type=float, default=np.power(0.01, 2.0)
    )
    debug_groups.add_argument(
        "--sim_meas_cov_val_z", type=float, default=np.power(0.01, 2.0)
    )

    args = parser.parse_args()

    np.set_printoptions(linewidth=2000)

    logging.info("Program options:")
    logging.info(pprint(vars(args)))
    # run filter
    data_list = os.path.join(args.root_dir, "test_list.txt")
    with open(data_list) as f:
        data_names = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    param_dict = vars(args)
    param_dict["date"] = str(datetime.datetime.now())
    with open(args.out_dir + "/parameters.json", "w") as parameters_file:
        parameters_file.write(json.dumps(param_dict, indent=4, sort_keys=True))

    # load offline calibration for IMU
    if args.dataset_number is not None:
        logging.info("Running in one-shot mode")
        logging.info("Using dataset {}".format(data_names[args.dataset_number]))
        trackerRunner = ImuTrackerRunner(args, data_names[args.dataset_number])
        trackerRunner.run_tracker(args)
    else:
        logging.info("Running in batch mode")
        # add metadata for logging
        n_data = len(data_names)
        for i, name in enumerate(data_names):
            logging.info(f"Processing {i} / {n_data} dataset {name}")
            try:
                trackerRunner = ImuTrackerRunner(args, name)
                trackerRunner.run_tracker(args)
            except FileExistsError as e:
                print(e)
                continue
            except OSError as e:
                print(e)
                continue
