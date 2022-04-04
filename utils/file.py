import os
import shutil
import sys

import torch

def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        option = input('Path %s already exists. Delete[d], Terminate[*]? ' % path)
        if option.lower() == 'd':
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('Terminated.')
            sys.exit(2)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))