import numpy as np
import matplotlib.pyplot as plt
import os
path = '././output/'

def calculate_nn_metrics(exp_list, output_file_name): #aug_list, 

    # exp_list = ['TLIO-master/output/eq_frame_2scalars_so2_2vec/aria_test',
    #             'TLIO-master/output/eq_frame_fullcov_so2_2vec/aria_test',
    #             'TLIO-master/output/eq_frame_pearsoncov_so2_2vec/aria_test'
    #             ]

    # aug_list = ['test_list.txt']
    result = []
    delta = 200 # depends on how long the chunking we want- 20 for 1s, 20*60 for 1 min because sample_freq was 20Hz

    for dir_path in exp_list:
        print('results for '+dir_path)
        # print(os.getcwd())
        # path = 'TLIO-master/output/'
        # dir_path = exp+'/aria_test'
        avg_list = []
        mse_list = []
        mse_x = []
        mse_y = []
        mse_z = []
        avg_mse = []
        final_avg = []
        final_median = []
        ate_list = []
        rte_list = []
        final_ate = []
        final_rte = []
        # for aug in aug_list:
        #     print(aug)
            # data_list=[]
            # with open('TLIO-master/local_data/tlio_golden/'+aug) as f:
            #     data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
            
            # for folder in  data_list: 
        for folder in os.listdir(dir_path):
            if os.path.isfile(dir_path+'/'+folder):
                print(folder, 'does not exist')
                continue
            else:
                if os.path.exists(dir_path+'/'+folder+'/net_outputs.txt'):
                    local_g_test =np.loadtxt(dir_path+'/'+folder+'/net_outputs.txt', delimiter=",")
                    
                    pred = local_g_test[:,1:4]
                    targs = local_g_test[:,4:7]
                    traj = np.loadtxt(dir_path+'/'+folder+'/trajectory.txt', delimiter=",") 
                    dts = np.mean(traj[1:,0] - traj[:-1,0])
                    pred_traj = np.cumsum(pred[:, :] *dts, axis=0) + traj[0,-3:]
                    targ_traj = np.cumsum(targs[:, :] *dts, axis=0) + traj[0,-3:]
                    ate = np.mean(np.linalg.norm(pred_traj - targ_traj[:,-3:], axis=1))
                    
                    re_est = pred_traj[1:-1-delta:delta] - pred_traj[delta:-1:delta]
                    re_gt = targ_traj[1:-1-delta:delta] - targ_traj[delta:-1:delta]

                    if 'rnin-vio' in dir_path:
                        ate_list.append(ate)
                        re_est = pred_traj[1:-1-16:16] - pred_traj[16:-1:16][:pred_traj[1:-1-16:16].shape[0]] #because seq len is 16 for rnin-vio
                        re_gt = targ_traj[1:-1-16:16] - targ_traj[16:-1:16][:targ_traj[1:-1-16:16].shape[0]]
                        rte_list.append(np.sqrt(np.mean(np.linalg.norm(re_gt - re_est, axis=1)** 2)))
                    else:
                        ate_list.append(ate)
                        rte_list.append(np.sqrt(np.mean(np.linalg.norm(re_gt - re_est, axis=1)** 2)))

                    error = np.linalg.norm(local_g_test[:, 4:7] - local_g_test[:,1:4], axis=1)**2
                    avg_list.append(error.mean())
                    error = np.mean((local_g_test[:, 4:7] - local_g_test[:,1:4]) ** 2, axis=0)
                    mse_x.append(error[0])
                    mse_y.append(error[1])
                    mse_z.append(error[2])
                    avg_mse.append(np.mean(error))
            # final_avg.append(np.array(avg_list).mean())
            # final_median.append(np.median(np.array(np.array(avg_list))))
            final_rte.append(np.array(ate_list).mean())
            final_ate.append(np.array(rte_list).mean())

        result.append({'exp' :dir_path,'avg':np.array(avg_list).mean(),'median':np.median(np.array(avg_list)),
                   'std':np.array(avg_list).std(),'avg_mse_x':np.array(mse_x).mean(),'avg_mse_y':np.array(mse_y).mean(),
                   'avg_mse_z':np.array(mse_z).mean(),'avg_mse_loss':np.array(avg_mse).mean(),'avg_ate':np.array(ate_list).mean(),
                   'avg_rte':np.array(rte_list).mean(), 'median_ate':np.median(np.array(ate_list)),'median_rte': np.median(np.array(rte_list))})

    import pandas as pd
    result_df = pd.DataFrame(result) 
    result_df.to_csv(output_file_name+'.csv') 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        type=str,
        action='append',  # Allows multiple uses of the argument
        # required=True,
        help="A file to process (can be used multiple times)",
        default=['/home/royinakj/TLIO-master/output/tlio_ev_se3p/nn_test']
    )
    # parser.add_argument("--test_seq_list", type=str, default="test_list.txt")
    parser.add_argument("--output_file_name", type=str, default="tlio_aug")

    args = parser.parse_args()

    # exp_list,gt_path,output_file_name
    calculate_nn_metrics(args.files, args.output_file_name) #args.test_seq_list, 