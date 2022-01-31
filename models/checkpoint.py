
import os
import torch


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pth.tar')):
        raise ValueError("Model pretrain path {} does not exists.".format(path))
    param_state_dict =torch.load(path + ".pth.tar")
    model.load_state_dict(param_state_dict)
    return
