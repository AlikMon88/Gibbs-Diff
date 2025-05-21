import numpy as np
import math
import os
import torch
import sys

class DCNN():
    def __init__(self, git_path = './KAIR'):
        self.git_path = git_path
        sys.path.append(kair_dir)

    def call():
        from models.network_dncnn import DnCNN as net
        model_path = os.path.join(self.git_path, 'model_zoo', 'dncnn_color_blind.pth') # This would need to be downloaded from the KAIR repository
        model_dncnn = net(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
        model_dncnn.load_state_dict(torch.load(model_path), strict=True)
        model_dncnn.eval();
        for k, v in model_dncnn.named_parameters():
            v.requires_grad = False
        model_dncnn = model_dncnn.to(device)

        return model_cnn



if __name__ == '__main__':
    print('__DCNN__.py__')