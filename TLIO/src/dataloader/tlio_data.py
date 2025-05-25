"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

"""
This file includes the main libraries in the network training module.
"""

import os
import time
from itertools import repeat
import torch
import numpy as np

#from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


from utils.logging import logging
from .constants import DatasetGenerationParams
from .memmapped_sequences_dataset import MemMappedSequencesDataset
from .iterable_pseudorandom_sequences_dataset import IterablePseudoRandomSequencesDataset
from .data_transform import TransformAddNoiseBias, TransformPerturbGravity, TransformInYawPlane

log = logging.getLogger(__name__)

def custom_collate_3d_pad_beginning(batch):
    """
    Custom collate function for batching 3D tensors with variable last dimensions (channels).
    Pads the last dimension at the beginning of the tensor.

    Args:
        batch (list of torch.Tensor): A list of 3D tensors with shapes [D, H, C],
                                      where the last dimension (C) varies.

    Returns:
        padded_batch (torch.Tensor): A tensor of shape [batch_size, D, H, max_channels], 
                                     where max_channels is the maximum size of the last dimension in the batch.
        channel_sizes (torch.Tensor): A tensor containing the original sizes of the last dimension (C) for each sample.
    
        
     structure of sample:
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
    """
    # Determine the maximum size of the last dimension (channels)
    max_channels = max(sample['feats']['imu0'].shape[1] for sample in batch)

    ts_us_all = []
    targ_dR_World_all = []
    targ_dt_World_all = []
    vel_World_all = []
    R_world_gla_all = []
    rot = []
    pos = []
    no_events = []
    feat_o2_all = []
    # Initialize a list to hold the padded tensors
    padded_batch = []
    pe_ts_all = []

    for sample in batch:
        # Compute the padding size for the beginning of the last dimension
        padding_size = max_channels - sample['feats']['imu0'].shape[1]
        
        # Pad the tensor on the left side (start of the last dimension)
        # Padding format for F.pad is (right_pad, left_pad) for each dimension.
        padded_sample = np.concatenate([np.zeros((6,padding_size)),sample['feats']['imu0']], axis=1)# Only pad the last dimension
        padded_batch.append(padded_sample)
        padded_sample_o2 = np.concatenate([np.zeros((9,padding_size)),sample['feats']['feat_o2']], axis=1)
        feat_o2_all.append(padded_sample_o2)
        ts_us_all.append(sample['ts_us'])
        targ_dR_World_all.append(sample['targ_dR_World'])
        pe_ts_all.append(sample['feats']['pe_ts'])
        targ_dt_World_all.append(sample['targ_dt_World'])
        vel_World_all.append(sample['vel_World'])
        R_world_gla_all.append(sample['R_world_gla'])
        rot.append(sample['rot'])
        pos.append(sample['pos'])
        no_events.append(sample['no_events'])

    # Stack the padded tensors along a new batch dimension
    padded_batch = torch.Tensor(np.stack(padded_batch, axis=0))
    ts_us_all = torch.Tensor(np.stack(ts_us_all, axis=0))
    targ_dR_World_all = torch.Tensor(np.stack(targ_dR_World_all, axis=0))
    feat_o2_all = torch.Tensor(np.stack(feat_o2_all, axis=0))
    targ_dt_World_all = torch.Tensor(np.stack(targ_dt_World_all, axis=0))
    vel_World_all =  torch.Tensor(np.stack(vel_World_all, axis=0))
    R_world_gla_all = torch.Tensor(np.stack(R_world_gla_all, axis=0))
    rot = torch.Tensor(np.stack(rot, axis=0))
    pos = torch.Tensor(np.stack(pos, axis=0))
    pe_ts_all = torch.Tensor(np.stack(pe_ts_all, axis=0))

    feats={}
    feats['pe_ts'] = pe_ts_all
    feats['imu0'] = padded_batch
    feats['feat_o2'] = feat_o2_all
    batched_sample = {
                "ts_us": ts_us_all,
                "feats": feats,
                "targ_dR_World": targ_dR_World_all,
                "targ_dt_World": targ_dt_World_all,
                "vel_World": vel_World_all,
                "R_world_gla": R_world_gla_all,
                "rot" : rot,
                "pos" : pos,
                "no_events": torch.Tensor(no_events),
            }


    return batched_sample

class TlioData:
    def __init__(
        self,
        data_path,
        batch_size=1,
        num_workers=1,
        window_size = 200,
        persistent_workers=True,
        only_n_sequence=-1,
        task_subset=None,
        ignore_tasks=None,
        decimator=10,
        dataset_style="mmap", # "mmap", "ram", or "iter". "iter" is best for huge datasets but sacrifice true randomness, mmap can go a bit farther than "ram" which just stores all in memory
        data_window_config={
            "window_size": 200, # 200 window size @200 Hz for 1sec of input data
            "step_period_us": 5000, # NOTE: unused at this point
            "data_in_local_gravity_aligned": True,
            "data_in_local_frame": False,
            "input_sensors": ["imu0"],
            "data_style": "resampled",
            "g_compensate" : False,
        },
        augmentation_options={   
            "do_bias_shift": True,
            "bias_shift_options": {
                "accel_bias_range": 0.2,#0.2
                "gyro_bias_range": 0.05, #0.05
                "accel_noise_std": 0,
                "gyro_noise_std": 0,
                "mag_bias_range": 0.05, # 0.05 In Gauss (.25-.65 Gauss is normal on earth)
                "barom_press_bias_range": 0.01, # 0.01 In Pascals (always near 1.0)
                "barom_temp_bias_range": 1, # 1 In deg celcius
            },
            "perturb_gravity": True,
            "perturb_gravity_theta_range": 5.0,#5.0
            "yaw_augmentation" : True,
        },
        event_based_input = False,
        interpolate = False,
        interp_freq = 200,
        base_freq = 200,
        base_event_stack = False,
        geodesic_event = False,
        rot_component_weight = 2,
        contrast_threshold = 0.01,
        add_vel_perturb = False,
        add_vel_perturb_range = 0.2,
        se3_events = False,
        polarity_input = False,
        imu_channel_freq = 200.0,
        polarity_noise_range = 0.0,
        noise_before_event_gen = False,
        arch = 'resnet'
    ):
        super().__init__()

        self.batch_size = batch_size
        self.data_path = data_path
        self.data_window_config = data_window_config
        self.data_window_config['window_size'] = window_size
        self.augmentation_options = augmentation_options
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.only_n_sequence = only_n_sequence
        if task_subset == []:
            task_subset = None
        if ignore_tasks == []:
            ignore_tasks = None
        self.task_subset = task_subset
        self.ignore_tasks = ignore_tasks
        self.decimator = decimator
        self.polarity_noise_range = polarity_noise_range

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.transform_done_in_dataloader = False
        self.dataset_style = dataset_style

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
        self.arch = arch
        self.noise_before_event_gen = noise_before_event_gen

    #def setup(self, stage=None):

    def prepare_data(self, testing=False):
        def setup_split(split):
            start_t = time.time()
            log.warning(
                f"{split}_dataloader : data_window_config is partially ignored here for now! "
                "(past and future data should be 0 for now)"
            )
            starting_point_time_us = 0  # TODO(dcaruso) positive if past imu data here
            prediction_times_us = 0  # TODO(dcaruso) negative if future imu data here
            genparams = DatasetGenerationParams(
                window_size=self.data_window_config["window_size"],
                step_period_us=self.data_window_config["step_period_us"],
                prediction_times_us=[prediction_times_us],
                starting_point_time_us=starting_point_time_us,
                generate_data_period_us=self.data_window_config["step_period_us"],
                decimator=self.decimator,
                express_in_local_gravity_aligned=self.data_window_config[
                    "data_in_local_gravity_aligned"
                ],
                input_sensors=self.data_window_config["input_sensors"],
                data_style=self.data_window_config["data_style"],
                express_in_local_frame = self.data_window_config["data_in_local_frame"],
                g_compensate = self.data_window_config['g_compensate']
            )
            
            if self.dataset_style == "mmap":
                SequencesDataset = MemMappedSequencesDataset
            elif self.dataset_style == "ram":
                SequencesDataset = lambda *args, **kwargs: MemMappedSequencesDataset(*args, **kwargs, store_in_ram=True)
            elif self.dataset_style == "iter":
                SequencesDataset = IterablePseudoRandomSequencesDataset
            else:
                raise ValueError(f"Unknown dataset_style \"{self.dataset_style}\"")

            dataset = SequencesDataset(
                self.data_path,
                split,
                genparams,
                only_n_sequence=self.only_n_sequence,
                verbose=True,
                event_based_input = self.event_based_input,
                interpolate = self.interpolate,
                interp_freq = self.interp_freq,
                base_freq = self.base_freq,
                base_event_stack = self.base_event_stack,
                geodesic_event = self.geodesic_event,
                rot_component_weight = self.rot_component_weight,
                contrast_threshold = self.contrast_threshold,
                add_vel_perturb = self.add_vel_perturb,
                add_vel_perturb_range = self.add_vel_perturb_range,
                se3_events = self.se3_events,
                polarity_input = self.polarity_input,
                imu_channel_freq = self.imu_channel_freq,
                gyro_bias_range = self.augmentation_options['bias_shift_options']['gyro_bias_range'],
                accel_bias_range = self.augmentation_options['bias_shift_options']['accel_bias_range'],
                theta_range_deg = self.augmentation_options['perturb_gravity_theta_range'],
                perturb_gravity = self.augmentation_options['perturb_gravity'],
                polarity_noise_range = self.polarity_noise_range,
                noise_before_event_gen = self.noise_before_event_gen,
                arch = self.arch
            )   
            
            setattr(self, f"{split}_dataset", dataset)
            end_t = time.time()
            log.info(f"{split} set loaded. Loading time: {end_t - start_t:.3f}s")
            #log.info(f"Number of {split} samples: {len(dataset)}")
        
        if testing:
            setup_split("test")
        else:
            for split in ["val", "train"]:
                setup_split(split)


    def train_dataloader(self):
        """
        # Make train and val the same if doing quick dev run
        if self.only_n_sequence > 0:
            log.warning(
                f"\nSwapping train dataset for val dataset for fast dev run "
                f"with sequences {list(self.val_dataset.data_list)}\n"
            )
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=True,
            )
        else:
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle="iter" not in self.dataset_style,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        # If no test split was set (e.g., dev dataset), just return val split
        if len(self.test_dataset) > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=True,
            )
        else:
            log.warning("Test set has no data. Returning validation set for testing")
            return self.val_dataloader()

    def get_train_transforms(self):
        transforms = []
        if self.augmentation_options["do_bias_shift"]:
            transforms.append(
                TransformAddNoiseBias(self.data_window_config["input_sensors"],
                    **self.augmentation_options["bias_shift_options"], polarity_input=self.polarity_input, contrast_threshold=self.contrast_threshold)
            )

        if self.augmentation_options["perturb_gravity"]:
            transforms.append(
                TransformPerturbGravity(self.data_window_config["input_sensors"], 
                    self.augmentation_options["perturb_gravity_theta_range"], polarity_input=self.polarity_input)
            )
        if self.augmentation_options['yaw_augmentation'] or self.augmentation_options['scale_augmentation'] or self.augmentation_options['reflection_yaw_augmentation']:
            transforms.append(TransformInYawPlane(self.data_window_config["input_sensors"], 
                                                  yaw_augmentation = self.augmentation_options['yaw_augmentation'],
                                                  scale_augmentation=self.augmentation_options['scale_augmentation'],
                                                  reflection_yaw_augmentation = self.augmentation_options['reflection_yaw_augmentation'],
                                                  interp_freq=self.augmentation_options['interp_freq'], polarity_input=self.polarity_input))
        return transforms

    def get_datalist(self, split="val"):
        dataset = getattr(self, f"{split}_dataset")
        assert dataset is not None, f"Tried to get {split} list but {split}_dataset is None"
        return dataset.data_list

