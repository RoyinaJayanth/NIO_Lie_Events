"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import math
import torch
#from pytorch3d.transforms import so3_exponential_map
from utils.torch_math_utils import so3_exp_map
from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

# TODO augs for mag and barom
class TransformAddNoiseBias:
    def __init__(self, input_sensors, 
            accel_noise_std, gyro_noise_std, accel_bias_range, gyro_bias_range, 
            mag_bias_range, barom_press_bias_range, barom_temp_bias_range, polarity_input=False, contrast_threshold=0.05):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_range = gyro_bias_range
        self.accel_bias_range = accel_bias_range
        self.mag_bias_range = mag_bias_range
        self.barom_press_bias_range = barom_press_bias_range
        self.barom_temp_bias_range = barom_temp_bias_range
        self.input_sensors = input_sensors
        self.polarity_input = polarity_input
        # self.polarity_shift_range = 1.0
        self.contrast_threshold = contrast_threshold

    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]
            
        for sensor, feat in feats_new.items():
            N, _InDim, T = feat.shape
            feat_aug = feat.clone()
            if "imu" in sensor:
                if self.polarity_input:
                    assert feat.shape[1] == 12
                    # feat_aug[:, 6:12, :] += (
                    #     (torch.rand(N, 6, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    #     * self.polarity_shift_range / 0.5
                    # )
                    # feat_aug[:, 6:12, :] = (feat_aug[:, 6:12, :]/(torch.norm(feat_aug[:, 6:12, :],dim=-2, keepdim=True)))*self.contrast_threshold
                else:
                    assert feat.shape[1] == 6
                # shift in the accel and gyro bias terms
                ## mask for event based inputs - augment only the non-zero elements
                mask = (feat_aug != 0).int()
                # mask = torch.ones_like(feat_aug)
                feat_aug[:, :3, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.gyro_bias_range / 0.5
                ) * mask[:, :3, :]
                feat_aug[:, 3:6, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.accel_bias_range / 0.5
                ) * mask[:, 3:6, :]

                # add gaussian noise
                feat_aug[:, :3, :] += (
                    torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std
                ) * mask[:, :3, :]
                feat_aug[:, 3:6, :] += (
                    torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                ) * mask[:, 3:6, :]

            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.mag_bias_range / 0.5
                )
                
                # TODO gaussian noise? 

            elif "barom" in sensor:
                assert feat.shape[1] == 2
                feat_aug[:, 0:1, :] += (
                    (torch.rand(N, 1, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.barom_press_bias_range / 0.5
                )
                feat_aug[:, 1:2, :] += (
                    (torch.rand(N, 1, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.barom_temp_bias_range / 0.5
                )

            # else:
                # print('Not augmenting ', sensor)
                # assert False
            
            feats_new[sensor] = feat_aug
        
        sample_new = {k: v for k,v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        return sample_new

class TransformPerturbGravity:
    def __init__(self,  input_sensors, theta_range_deg, polarity_input=False):
        self.theta_range_deg = theta_range_deg
        self.input_sensors = input_sensors
        self.polarity_input = polarity_input

    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]

        # get rotation vector of random horizontal direction
        angle_rand_rad = (
            torch.rand(sample["ts_us"].shape[0], device=sample["ts_us"].device, dtype=torch.float32) * math.pi * 2
        )
        theta_rand_rad = (
            torch.rand(sample["ts_us"].shape[0], device=sample["ts_us"].device, dtype=torch.float32)
            * math.pi
            * self.theta_range_deg
            / 180.0
        )
        c = torch.cos(angle_rand_rad)
        s = torch.sin(angle_rand_rad)
        zeros = torch.zeros_like(angle_rand_rad)
        vec_rand = torch.stack([c, s, zeros], dim=1)
        rvec = theta_rand_rad[:, None] * vec_rand  # N x 3
        R_mat = so3_exp_map(rvec)  # N x 3 x 3

        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
                if self.polarity_input:
                    assert feat.shape[1] == 12
                    feat_aug[:, 6:9, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 6:9, :])
                    feat_aug[:, 9:12, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 9:12, :])
                else:
                    assert feat.shape[1] == 6
            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
            elif "barom" in sensor:
                # Nothing to do here
                pass
            # else:
            #     # assert False
            #     print('Not augmenting ', sensor)
            feats_new[sensor] = feat_aug
        
        sample_new = {k: v for k,v in sample.items() if k != "feats"}

        sample_new["feats"] = feats_new
        return sample_new


class TransformInYawPlane:
    """this transform object:
        - rotate imu data in horizontal plane with a random planar rotation
        - rotate the target the same way
    this brings invariance in the data to planar rotation
    this can also prevent the network to learn cues specific to the IMU placement
    """

    def __init__(self, input_sensors,yaw_augmentation, scale_augmentation,reflection_yaw_augmentation, angle_half_range_rad=math.pi, interp_freq =200, polarity_input = False):
        """
        Random yaw angles will be in [-angle_half_range, angle_half_range]
        """
        self.input_sensors = input_sensors
        self.angle_half_range_rad = angle_half_range_rad
        self.reflections = reflection_yaw_augmentation
        self.scale_aug = scale_augmentation
        self.rot_aug = yaw_augmentation
        self.ind_interp_scale = scale_augmentation
        self.interp_freq = interp_freq
        self.base_freq = 200
        self.polarity_input = polarity_input

    def __call__(self, sample):

        if self.scale_aug: # can work only for tlio because discarding pe_ts and o_2 features
            device = sample['ts_us'].device
            
            ## discrete imu frequencies
            freq = [40,50,100,200]#[100,200,160,400,800][100,125,200,250,500,1000][40,50,100,200]
            imu_freq = freq[torch.randperm(4)[0]]
            self.interp_freq = imu_freq
            sample_scale = { 
                "seq_id" : sample['seq_id'],
                "ts_us": sample['ts_us'].flip(1)[:,::int(self.base_freq/imu_freq)].flip(1),#[::-int(self.base_freq/imu_freq)][::-1]
                "targ_dR_World": sample['targ_dR_World'],
                "targ_dt_World": sample['targ_dt_World'],
                "vel_World": sample['vel_World'],
                "R_world_gla": sample['R_world_gla'],
                "imu_freq" : imu_freq,
                "rot":sample['rot'],
                "pos":sample['pos'],
                "no_events":sample["no_events"]
            }
            sample_scale["feats"] ={}
            sample_scale["feats"]['imu0']=sample["feats"]['imu0'].flip(2)[:,:,::int(self.base_freq/imu_freq)].flip(2)
            
        if self.ind_interp_scale and self.scale_aug==True:
            device = sample['ts_us'].device
            
            imu_freq = self.interp_freq
            old_ts = sample_scale['ts_us'].cpu().numpy()
            ts = np.zeros((old_ts.shape[0],self.base_freq,old_ts.shape[-1]))
            old_feat = sample_scale['feats']['imu0'].cpu().numpy()
            feats = np.zeros((old_feat.shape[0],old_feat.shape[1],self.base_freq))
            ## in the other version interpolate targ_dt and targ_dR and all feats without any of the einsums
            gyro = old_feat[:,:3]
            accel = old_feat[:,3:]
            for i in range(old_ts.shape[0]):#old_ts.shape[0]
                ts[i] = np.linspace(old_ts[i,0], old_ts[i,-1], self.base_freq, endpoint=True).reshape((-1,1))
                # print(old_ts[i,-1], ts[i,-1], imu_freq)
                assert old_ts[i,-1] == ts[i,-1]
                feats[i,:3] = interpolate.interp1d(old_ts[i].reshape((-1)), gyro[i], axis=-1)(ts[i].reshape((-1)))
                feats[i,3:] = interpolate.interp1d(old_ts[i].reshape((-1)), accel[i], axis=-1)(ts[i].reshape((-1)))

              
            
            # ignoring pe_ts and o_2
            sample_scale = { 
                "seq_id" : sample['seq_id'],
                "ts_us":torch.Tensor(ts).to(device),
                "targ_dR_World": sample['targ_dR_World'],
                "targ_dt_World": sample['targ_dt_World'],
                "vel_World": sample['vel_World'],
                "R_world_gla": sample['R_world_gla'],
                "imu_freq" : imu_freq,
                "rot":sample['rot'],
                "pos":sample['pos'],
                "no_events":sample["no_events"]
            }
            sample_scale["feats"] ={}
            sample_scale["feats"]['imu0']=torch.Tensor(feats).to(device)

        if self.ind_interp_scale and self.scale_aug==False:
            device = sample['ts_us'].device
            
            imu_freq = self.interp_freq
            old_ts = sample['ts_us'][:,::int(self.base_freq/imu_freq)].cpu().numpy()
            ts = np.zeros((old_ts.shape[0],self.base_freq,old_ts.shape[-1]))
            old_feat = sample['feats']['imu0'][:,:,::int(self.base_freq/imu_freq)].cpu().numpy()
            feats = np.zeros((old_feat.shape[0],old_feat.shape[1],self.base_freq))
            pos = sample['pos'][:,::int(self.base_freq/imu_freq)].cpu().numpy()
            targ_pos = np.zeros((pos.shape[0],self.base_freq,pos.shape[-1]))
            vel = sample['vel_World'][:,::int(self.base_freq/imu_freq)].cpu().numpy()
            targ_vel = np.zeros((vel.shape[0],self.base_freq,vel.shape[-1]))
            R_w = sample['rot'][:,::int(self.base_freq/imu_freq)].cpu().numpy()
            targ_R_w = np.zeros((R_w.shape[0], self.base_freq,R_w.shape[-2],R_w.shape[-1]))
            ## in the other version interpolate targ_dt and targ_dR and all feats without any of the einsums
            R_i = np.einsum('ntji,njk->ntik', R_w, sample['R_world_gla'].cpu().numpy())
            gyro = np.einsum('ntik,nkt->nit',R_i,old_feat[:,:3])
            accel = np.einsum('ntik,nkt->nit',R_i,old_feat[:,3:])
            for i in range(old_ts.shape[0]):#old_ts.shape[0]
                ts[i] = np.linspace(old_ts[i,0], old_ts[i,-1], self.base_freq, endpoint=True).reshape((-1,1))
                # print(old_ts[i,-1], ts[i,-1], imu_freq)
                assert old_ts[i,-1] == ts[i,-1]
                feats[i,:3] = interpolate.interp1d(old_ts[i].reshape((-1)), gyro[i], axis=-1)(ts[i].reshape((-1)))
                feats[i,3:] = interpolate.interp1d(old_ts[i].reshape((-1)), accel[i], axis=-1)(ts[i].reshape((-1)))
                targ_pos[i]  = interpolate.interp1d(old_ts[i].reshape((-1)), pos[i], axis=0)(ts[i].reshape((-1)))
                targ_vel[i] = interpolate.interp1d(old_ts[i].reshape((-1)), vel[i], axis=0)(ts[i].reshape((-1)))

                ## slerp interpolation for orientation and gravity alignment
                targ_R_slerp = Slerp(old_ts[i].reshape((-1)), Rotation.from_matrix(R_w[i]))
                targ_R_w[i] = targ_R_slerp(ts[i].reshape((-1))).as_matrix()
            ## convert imu back to gla
            R_i = np.einsum('njk,ntji->ntki', sample['R_world_gla'].cpu().numpy(),targ_R_w)
            feats[:,:3] = np.einsum('ntik,nkt->nit',R_i,feats[:,:3])
            feats[:,3:] = np.einsum('ntik,nkt->nit',R_i,feats[:,3:]) 
            ## get disp and relative orientation
            targ_R = np.einsum('ntij,nkj->ntik', targ_R_w, targ_R_w[:,0,:,:])#R_W_i @ R_W_0.transpose([0,2,1])
            targ_dt = targ_pos - targ_pos[:,0:1]
            targ_dt = np.einsum("nji,ntj->nti", sample['R_world_gla'].cpu().numpy(), targ_dt)
            # ignoring pe_ts and o_2
            sample_scale = { 
                "seq_id" : sample['seq_id'],
                "ts_us": torch.Tensor(ts).to(device),
                "targ_dR_World": torch.Tensor(targ_R.astype(np.float32)).to(device),
                "targ_dt_World": torch.Tensor(targ_dt.astype(np.float32)).to(device),
                "vel_World": torch.Tensor(targ_vel.astype(np.float32)).to(device),
                "R_world_gla": sample['R_world_gla'],
                "no_events" : sample["no_events"]
            }
            sample_scale["feats"] ={}
            sample_scale["feats"]['imu0']=torch.Tensor(feats).to(device)
        if self.scale_aug==False and self.ind_interp_scale==False:
            sample_scale = sample
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample_scale["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]
        # rotate in the yaw plane
        N = sample_scale["ts_us"].shape[0]
        rand_unif = 2*torch.rand((N), device=sample_scale["ts_us"].device, dtype=torch.float32) - 1 # in [-1,1]
        angle_rad = rand_unif * self.angle_half_range_rad
        c = torch.cos(angle_rad)  # N
        s = torch.sin(angle_rad)  # N
        ones = torch.ones_like(c)  # N
        zeros = torch.zeros_like(s)  # N
        R_newWorld_from_oldWorld_flat = torch.stack(
                (c, -s, zeros, s, c, zeros, zeros, zeros, ones), dim=1
            )  # N x 9
        R_newWorld_from_oldWorld = R_newWorld_from_oldWorld_flat.reshape((N, 3, 3))
        
    
        if self.reflections:
            indices = torch.randperm(N)[:int(0.5*N)]
            P = torch.eye(3).unsqueeze(0).expand(N,-1,-1).clone().to(sample_scale["ts_us"].device)
            P[indices] = P[indices]@ torch.Tensor([[0,1,0],[1,0,0],[0,0,1]]).to(sample_scale["ts_us"].device)
            R_newWorld_from_oldWorld = torch.einsum('nik,nkt->nit', R_newWorld_from_oldWorld, P)
            for sensor, feat in feats_new.items():
                feat_aug = feat.clone()
                if "imu" in sensor:
                    assert feat.shape[1] == 6
                    #feat_aug[indices, :3, :] = -1 * feat_aug[indices, :3, :] --just to experiment with wrong augmentations
                    feat_aug[:, :3, :] = torch.einsum(
                        "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                    )
                    feat_aug[:, 3:6, :] = torch.einsum(
                        "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, 3:6, :]
                    )
                    
                elif "mag" in sensor:
                    assert feat.shape[1] == 3
                    feat_aug[:, :3, :] = torch.einsum(
                        "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                    )
                elif "barom" in sensor:
                    # Nothing to do here
                    pass
                # else:
                #     assert False

                feats_new[sensor] = feat_aug

            sample_new = {
                k: v for k,v in sample_scale.items() 
                if k.split("second_")[-1].split("_same")[0] not in ["feats","targ_dt_World","vel_World"]
            }
            sample_new["feats"] = feats_new
            # Handle the target data. Only displacement and vel need rotating, not relative rotation (already relative).
            for k in "targ_dt_World", "vel_World":
                sample_new[k] = torch.einsum("nik,ntk->nti", R_newWorld_from_oldWorld, sample_scale[k])

            return sample_new
        elif self.rot_aug:
            for sensor, feat in feats_new.items():
                feat_aug = feat.clone()
                if "imu" in sensor:
                    
                    feat_aug[:, :3, :] = torch.einsum(
                        "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                    )
                    feat_aug[:, 3:6, :] = torch.einsum(
                        "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, 3:6, :]
                    )
                    if self.polarity_input:
                        assert feat.shape[1] == 12
                        feat_aug[:, 6:9, :] = torch.einsum(
                            "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, 6:9, :]
                        )
                        feat_aug[:, 9:12, :] = torch.einsum(
                            "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, 9:12, :]
                        )
                    else:
                        assert feat.shape[1] == 6
                elif "mag" in sensor:
                    assert feat.shape[1] == 3
                    feat_aug[:, :3, :] = torch.einsum(
                        "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                    )
                elif "barom" in sensor:
                    # Nothing to do here
                    pass
                # else:
                #     assert False

                feats_new[sensor] = feat_aug

            sample_new = {
                k: v for k,v in sample_scale.items() 
                if k.split("second_")[-1].split("_same")[0] not in ["feats","targ_dt_World","vel_World"]
            }
            sample_new["feats"] = feats_new
            # Handle the target data. Only displacement and vel need rotating, not relative rotation (already relative).
            for k in "targ_dt_World", "vel_World":
                sample_new[k] = torch.einsum("nik,ntk->nti", R_newWorld_from_oldWorld, sample_scale[k])

        
            return sample_new
        return sample_scale


"""
if __name__ == "__main__":
    # test
    def get_sample():
        acc = torch.tensor([[0], [1.0], [0]]).repeat(repeats=(1, 1, 200))
        gyr = torch.tensor([[0], [1.0], [0]]).repeat((1, 1, 200))
        Rarg = torch.tensor(torch.eye(3)).repeat(1, 1, 1)
        targ = torch.tensor([0, 1.0, 0]).repeat(1, 1)
        print(acc.shape)
        print(gyr.shape)
        print(Rarg)
        print(targ.shape)
        samples = (torch.cat((acc, gyr), dim=1), Rarg, targ)
        return samples

    transform = TransformInYawPlane("imu0")
    feat_aug, Rarg_aug, targ_aug = transform(
        get_sample(), angle_rad=torch.tensor(math.pi / 2)
    )

    print(feat_aug)
    print(Rarg_aug)
    print(targ_aug)
"""
