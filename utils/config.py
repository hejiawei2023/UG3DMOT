import os
import yaml
from easydict import EasyDict as edict


def load_config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')

    cfg = edict(yaml.safe_load(listfile1))
    cfg_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    for line in cfg_show:
        print(line)
    return cfg



