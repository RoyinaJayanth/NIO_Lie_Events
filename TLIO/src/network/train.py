"""
This file includes the main libraries in the network training module.
"""

import json
import os
import signal
import sys
import time
from functools import partial
from os import path as osp


import numpy as np
import torch
#from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from network.losses import get_loss
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.logging import logging
from utils.utils import to_device
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def torch_to_numpy(torch_arr):
    return torch_arr.detach().cpu().numpy()

def get_inference(network, data_loader, device, epoch, arch_type,scale_augmentation,fixed_val, transforms=[], only_polarity_input=False):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    events_list = []

    network.eval()

    for bid, sample in enumerate(tqdm(data_loader)):
        sample = to_device(sample, device)
        for transform in transforms: ## this is uncommented but just testing without this for scale augmentation
            sample  = transform(sample)

        feat = sample["feats"]["imu0"]
        if only_polarity_input:
            feat = feat[:,6:,:]
        pred, pred_cov = network(feat)
        
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)

        loss = get_loss(pred, pred_cov, targ, epoch, arch_type)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))
        events_list.append(torch_to_numpy(sample['no_events'].unsqueeze(-1)))
       

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    events_list = np.concatenate(events_list, axis=0)
    

   
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
        "events_list":events_list,
        "epoch" : epoch * np.ones_like(events_list),
    }
    print('targets shape:',attr_dict['targets'].shape)
    print('events shape:', attr_dict['events_list'].shape)
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, arch_type,scale_augmentation, transforms=[], only_polarity_input=False):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    events_list = []
    
    network.train()

    for bid, sample in enumerate(tqdm(train_loader)):
        # print('entered the loader embedding')
        sample = to_device(sample, device)
        # type(sample)
        for transform in transforms:
            sample = transform(sample)
        
        feat = sample["feats"]["imu0"]
        if only_polarity_input:
            feat = feat[:,6:,:]
        optimizer.zero_grad()
        pred, pred_cov = network(feat)

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        

        loss = get_loss(pred, pred_cov, targ, epoch, arch_type)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
        events_list.append(torch_to_numpy(sample['no_events'].unsqueeze(-1)))
            

        loss = loss.mean()
        loss.backward()

        
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=False)#0.1
        optimizer.step()
        
    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    events_list = np.concatenate(events_list, axis=0)
    

    
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
        "events_list":events_list,
        "epoch" : epoch * np.ones_like(events_list),
    }
    print('targets shape:',train_attr_dict['targets'].shape)
    print('events shape:', train_attr_dict['events_list'].shape)
    return train_attr_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """

    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)
    ml_loss = np.average(attr_dict["losses"])
    sigmas = np.exp(attr_dict["preds_cov"])
    # If it's sequential, take the last one
    if len(mse_loss.shape) == 2:
        assert mse_loss.shape[0] == 3
        mse_loss = mse_loss[:, -1]
        assert sigmas.shape[1] == 3
        sigmas = sigmas[:,:,-1]

    wandb.log({f"{mode}_loss/loss_x": mse_loss[0], f"{mode}_loss/loss_y": mse_loss[1],
               f"{mode}_loss/loss_z": mse_loss[2], f"{mode}_loss/avg": np.mean(mse_loss), 
               f"{mode}_dist/loss_full": ml_loss},step=epoch)

    summary_writer.add_scalar(f"{mode}_loss/loss_x", mse_loss[0], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_y", mse_loss[1], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_z", mse_loss[2], epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_x", sigmas[:, 0], epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_y", sigmas[:, 1], epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_z", sigmas[:, 2], epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(
        f"{mode}: average ml loss: {ml_loss}, average mse loss: {mse_loss}/{np.mean(mse_loss)}"
    )


def save_model(args, epoch, network, optimizer, best, interrupt=False):
    if interrupt:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
    if best:
        model_path = osp.join(args.out_dir, "checkpoint_best.pt")        
    else:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")
    
    add_input = 0
    if args.ev_file_name is not None:
        add_input = 1

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)+ add_input) ,
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
            int(args.past_time * args.imu_channel_freq)
            + int(args.window_time * args.imu_channel_freq)+add_input
            + int(args.future_time * args.imu_channel_freq)
        )
        // 32
        + 1
    }


    return data_window_config, net_config


def net_train(args):
    """
    Main function for network training
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        #if args.train_list is None:
        #    raise ValueError("train_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not osp.isdir(osp.join(args.out_dir, "checkpoints")):
                os.makedirs(osp.join(args.out_dir, "checkpoints"))
            if not osp.isdir(osp.join(args.out_dir, "logs")):
                os.makedirs(osp.join(args.out_dir, "logs"))
            with open(
                os.path.join(args.out_dir, "parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        #if args.val_list is None:
        #    logging.warning("val_list is not specified.")
        if args.continue_from is not None:
            if osp.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)

    train_loader, val_loader = None, None
    start_t = time.time()

    augmentation_options={   
            "do_bias_shift": args.do_bias_shift,
            "bias_shift_options": {
                "accel_bias_range": args.accel_bias_range,#0.2
                "gyro_bias_range": args.gyro_bias_range, #0.05
                "accel_noise_std": 0,
                "gyro_noise_std": 0,
                "mag_bias_range": 0.05, # 0.05 In Gauss (.25-.65 Gauss is normal on earth)
                "barom_press_bias_range": 0.01, # 0.01 In Pascals (always near 1.0)
                "barom_temp_bias_range": 1, # 1 In deg celcius
            },
            "perturb_gravity": args.perturb_gravity,
            "perturb_gravity_theta_range": args.perturb_gravity_theta_range,#5.0
            "yaw_augmentation" : args.yaw_augmentation,
            "scale_augmentation" : args.scale_augmentation,
            "reflection_yaw_augmentation" : args.reflection_yaw_augmentation,
            "interp_freq": args.interp_freq
        }
    data = TlioData(
        args.root_dir, 
        batch_size=args.batch_size, 
        dataset_style=args.dataset_style, 
        num_workers=args.workers,
        persistent_workers=args.persistent_workers,
        window_size = data_window_config['window_size'],
        augmentation_options = augmentation_options,
        event_based_input = args.event_based_input,
        interpolate = args.test_interpolate,
        interp_freq = args.interp_freq,
        base_freq = args.imu_freq,
        base_event_stack = args.base_event_stack,
        geodesic_event = args.geodesic_event,
        rot_component_weight = args.rot_component_weight,
        contrast_threshold = args.contrast_threshold,
        add_vel_perturb = args.add_vel_perturb,
        add_vel_perturb_range = args.add_vel_perturb_range,
        se3_events = args.se3_events,
        polarity_input = args.polarity_input,
        imu_channel_freq = args.imu_channel_freq,
        polarity_noise_range = args.polarity_noise_range,
        noise_before_event_gen = args.noise_before_event_gen,
        arch = args.arch
    )
    data.prepare_data()
    
    train_list = data.get_datalist("train")

    """
    try:
        train_dataset = FbSequenceDataset(
            args.root_dir, train_list, args, data_window_config, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
    except OSError as e:
        logging.error(e)
        return
    """
    train_loader = data.train_dataloader()
    train_transforms = data.get_train_transforms()

    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(data.train_dataset)}")

    #if args.val_list is not None:
    if data.val_dataset is not None:
        val_list = data.get_datalist("val")
        """
        try:
            val_dataset = FbSequenceDataset(
                args.root_dir, val_list, args, data_window_config, mode="val"
            )
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        """
        val_loader = data.val_dataloader()
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(data.val_dataset)}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim)
    network.to(device)
    total_params = network.get_num_params()
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from is not None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(osp.join(args.out_dir, "logs"))
    wandb.log({"total_param":total_params})
    summary_writer.add_text("info", f"total_param: {total_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    #attr_dict = get_inference(network, train_loader, device, start_epoch, train_transforms)
    #write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    #if val_loader is not None:
    #    attr_dict = get_inference(network, val_loader, device, start_epoch)
    #    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val")

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, best=False, interrupt=True)
        sys.exit()

    best_val_loss = np.inf
    all_mse_loss = []
    all_mse_loss_x = []
    all_mse_loss_y = []
    all_mse_loss_z = []
    all_ml_loss = []
    sigmas_after_training = np.empty(0)
    all_mse_loss_val = []
    all_mse_loss_x_val = []
    all_mse_loss_y_val = []
    all_mse_loss_z_val = []
    all_ml_loss_val = []
    sigmas_after_training_val = np.empty(0)
    train_loss = []
    val_loss = []
    train_epochs = []
    val_epochs = []
    all_events_list = []
    all_events_list_val = []
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, args.arch,args.scale_augmentation, train_transforms, only_polarity_input=args.only_polarity_input)
        train_loss.append(train_attr_dict['losses'])
        train_epochs.append(train_attr_dict['epoch'])

        mse_loss = np.mean((train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2, axis=0)
        ml_loss = np.average(train_attr_dict["losses"])
        
        all_mse_loss_x.append(mse_loss[0])
        all_mse_loss_y.append(mse_loss[1])
        all_mse_loss_z.append(mse_loss[2])
        all_mse_loss.append(np.mean(mse_loss))
        all_ml_loss.append(ml_loss)
        all_events_list.append(train_attr_dict['events_list'])

        print('train mse loss for ', epoch, ' is ', np.mean(mse_loss), ' and ml is ', ml_loss)
        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")
        save_model(args, epoch, network, optimizer, best=False)

        if val_loader is not None:
            if args.fixed_validation_set:
                val_attr_dict = get_inference(network, val_loader, device, epoch, args.arch,args.scale_augmentation, args.fixed_validation_set, only_polarity_input=args.only_polarity_input)#, train_transforms
            else:
                val_attr_dict = get_inference(network, val_loader, device, epoch, args.arch,args.scale_augmentation, args.fixed_validation_set, train_transforms, only_polarity_input=args.only_polarity_input)#, train_transforms
            val_loss.append(val_attr_dict['losses'])
            val_epochs.append(val_attr_dict['epoch'])

            mse_loss_val = np.mean((val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2, axis=0)
            ml_loss_val = np.average(val_attr_dict["losses"])
            all_mse_loss_x_val.append(mse_loss_val[0])
            all_mse_loss_y_val.append(mse_loss_val[1])
            all_mse_loss_z_val.append(mse_loss_val[2])
            all_mse_loss_val.append(np.mean(mse_loss_val))
            all_ml_loss_val.append(ml_loss_val)
            all_events_list_val.append(val_attr_dict['events_list'])

            print('val mse loss for ', epoch, ' is ', np.mean(mse_loss_val), ' and ml is ', ml_loss_val)

            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer, best=True)
                sigmas_after_training = np.exp(train_attr_dict["preds_cov"])
                sigmas_after_training_val = np.exp(val_attr_dict["preds_cov"])
        else:
            save_model(args, epoch, network, optimizer, best=False)
            sigmas_after_training = np.exp(train_attr_dict["preds_cov"])
            sigmas_after_training_val = np.exp(val_attr_dict["preds_cov"])

        # print('train shapes for storing:',np.array(all_mse_loss_x).shape, np.array(all_events_list).shape)
        ## saving the losses to plot graphs on training
        res_np_array = np.array([all_mse_loss_x, all_mse_loss_y, all_mse_loss_z, all_mse_loss, all_ml_loss])
        np.save(osp.join(args.out_dir, "training_losses.npy"),res_np_array.T)

        # print('val shapes for storing:', np.array(all_mse_loss_x_val).shape, np.array(all_events_list_val).shape)
        val_res_np_array = np.array([all_mse_loss_x_val, all_mse_loss_y_val, all_mse_loss_z_val, all_mse_loss_val, all_ml_loss_val])
        np.save(osp.join(args.out_dir, "val_losses_during_training.npy"),val_res_np_array.T)

        res_train_new = np.concatenate([np.concatenate(train_loss, axis=0), np.concatenate(all_events_list, axis=0), np.concatenate(train_epochs, axis=0)],axis=1)
        np.save(osp.join(args.out_dir, "train_loss_imu_freq.npy"),res_train_new)

        val_train_new = np.concatenate([np.concatenate(val_loss, axis=0), np.concatenate(all_events_list_val, axis=0), np.concatenate(val_epochs, axis=0)],axis=1)
        np.save(osp.join(args.out_dir, "val_loss_imu_freq.npy"),val_train_new)
        # print('loss imu shapes:', res_train_new.shape, val_train_new.shape)
        ## saving the sigmas for the training and val dataset corresponding to last checkpoint
        np.save(osp.join(args.out_dir, "sigmas_after_training.npy"),sigmas_after_training)
        np.save(osp.join(args.out_dir, "val_sigmas_after_training.npy"),sigmas_after_training_val)


    logging.info("Training complete.")

    return
