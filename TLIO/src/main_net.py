"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""

import network
from utils.argparse_utils import add_bool_arg
import os
import wandb
# os.add_dll_directory(os.getcwd())
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["WANDB__SERVICE_WAIT"] = "300"
import warnings
warnings.filterwarnings("ignore")

import random
import torch
import os
import numpy as np


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything()

    import argparse

    parser = argparse.ArgumentParser()

    # ------------------ directories -----------------
    # NOTE now they are assumed to be under root_dir with new format
    #parser.add_argument("--train_list", type=str, default=None)
    #parser.add_argument("--val_list", type=str, default=None)
    #parser.add_argument("--test_list", type=str, default=None) 
    parser.add_argument(
        "--root_dir", type=str, 
        default="../TLIO-master/local_data/tlio_golden", help="Path to data directory"
    )#/mnt/kostas-graid/datasets/royinakj/TLIO_events
    parser.add_argument("--out_dir", type=str, default="../TLIO-master/output/tlio_ev_se3p/nn_test")
    parser.add_argument("--model_path", type=str, default="../TLIO-master/output/tlio_ev_se3p/checkpoint_best.pt")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument(
        "--test_list", type=str, 
        default="test_list_original.txt", help="Path to test list" 
    )


    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04) 
    parser.add_argument("--batch_size", type=int, default=1024) #1024 --originally
    parser.add_argument("--epochs", type=int, default=3, help="max num epochs")

    parser.add_argument("--arch", type=str, default="resnet") 
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=12)# 6 if not using polarity and imu meas as input
    parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument("-j", "--workers", type=int, default=10)
    parser.add_argument("--dataset_style", type=str, default="mmap", 
            help="'ram', 'mmap', or 'iter'. See dataloader/tlio_data.py for more details")
    add_bool_arg(parser, "persistent_workers", default=True)

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="test", choices=["train", "test", "eval"]
    )
    parser.add_argument(
        "--imu_freq", type=float, default=200.0, help="imu_base_freq is a multiple"
    )
    parser.add_argument("--imu_base_freq", type=float, default=1000.0)
    parser.add_argument("--imu_channel_freq", type=float, default=200.0, help="input channels to resnet")

    # ----- perturbation -----
    add_bool_arg(parser, "do_bias_shift", default=False)
    parser.add_argument("--accel_bias_range", type=float, default=0.2)  # 5e-2 0.2 
    parser.add_argument("--gyro_bias_range", type=float, default=0.05)  # 1e-3 0.05

    add_bool_arg(parser, "perturb_gravity", default=False)
    parser.add_argument(
        "--perturb_gravity_theta_range", type=float, default=5.0
    )  # degrees 5.0
    add_bool_arg(parser, "yaw_augmentation", default=False)
    add_bool_arg(parser, "scale_augmentation", default=False)
    add_bool_arg(parser, "reflection_yaw_augmentation", default=False) 
    add_bool_arg(parser, "fixed_validation_set", default=True)
    parser.add_argument("--interp_freq",type=float, default=200.0)
    add_bool_arg(parser, "event_based_input", default=True)
    add_bool_arg(parser, "test_interpolate", default=False) 
    add_bool_arg(parser,"base_event_stack", default=False)
    add_bool_arg(parser,"geodesic_event",default=False)
    # higher rot_component_weight implies more rotation events
    parser.add_argument("--rot_component_weight",type=int, default=1)




    parser.add_argument("--contrast_threshold",type=float, default=0.01) #contrast_threshold 0.01
    add_bool_arg(parser,"add_vel_perturb",default=False)
    parser.add_argument("--add_vel_perturb_range",type=float, default=0.0) #add_vel_perturb default 0.2

    add_bool_arg(parser,"se3_events",default=True)

    add_bool_arg(parser,"noise_before_event_gen",default=False) #noise_before_event_gen
    add_bool_arg(parser,"gravity_noise_before_event_gen",default=False)
    add_bool_arg(parser,"init_vel_noise_sens",default=False)

    #polarity_input
    add_bool_arg(parser,"polarity_input",default=True)
    add_bool_arg(parser,"only_polarity_input",default=False)
    ## only while training - polarity_noise_range
    parser.add_argument("--polarity_noise_range",type=float, default=0.0)



    # ----- window size and inference freq -----
    parser.add_argument("--past_time", type=float, default=0.0)  # s
    parser.add_argument("--window_time", type=float, default=1.0)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    # sample freq is used to decide step size as int(args.imu_freq / args.sample_freq)
    ## 20 resuts in decimator of 10- or we skip 10 samples giving us update frequency of 20Hz
    parser.add_argument("--sample_freq", type=float, default=20.0)  # hz default=20.0 -they claim to use this in final system

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plot", default=False)
    parser.add_argument("--rpe_window", type=float, default="2.0")  # s default="2.0" ## this is only for plotting
    
    args = parser.parse_args()

#     run =   wandb.init(
#     # Set the project where this run will be logged
#     project="lie_events", config=args.__dict__, id=args.out_dir.split('/')[-1], resume="allow"
#     # Track hyperparameters and run metadata
# )

    ###########################################################
    # Main
    ###########################################################
    if args.mode == "train":
        network.net_train(args)
    elif args.mode == "test":
        network.net_test(args)
    elif args.mode == "eval":
        network.net_eval(args)
    else:
        raise ValueError("Undefined mode")
