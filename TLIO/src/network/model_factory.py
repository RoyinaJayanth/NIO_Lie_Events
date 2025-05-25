from network.model_resnet import BasicBlock1D, ResNet1D
from network.model_resnet_seq import ResNetSeq1D
from network.model_tcn import TlioTcn

from utils.logging import logging

from network.rnin_model_lstm import ResNetLSTMSeqNet
import yaml

def get_model(arch, net_config, input_dim=6, output_dim=3):
    
    if arch == "resnet_w_t": # 9 time positional encoding
        network = ResNet1D(
            BasicBlock1D, input_dim+9, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == 'rnin_vio_model_lstm':
        print('This is RNIN-VIO model!')
        with open('/home/royinakj/rnin-vio/TLIO_output_seqlen1/all_params.yaml', 'r') as f:
            rnin_cfg = yaml.load(f, Loader=yaml.Loader)
        network = ResNetLSTMSeqNet(rnin_cfg)
    elif arch == "resnet_bigger":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [3, 3, 3, 3], net_config["in_dim"]
        )
    elif arch == "resnet":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    # elif arch == "resnet": ## specifically testing polarity
    #     network = ResNet1D(
    #         BasicBlock1D, 6, output_dim, [2, 2, 2, 2], net_config["in_dim"]
    #     )
    elif arch == "resnet_seq":
        network = ResNetSeq1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "tcn":
        network = TlioTcn(
            input_dim,
            output_dim,
            [64, 64, 64, 64, 128, 128, 128],
            kernel_size=2,
            dropout=0.2,
            activation="GELU",
        )
    else:
        raise ValueError("Invalid architecture: ", arch)

    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    logging.info(f"Number of params for {arch} model is {num_params}")   

    return network
