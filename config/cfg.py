
from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()
_C.SEED = 42
_C.SAVE = False
_C.FIG = True
_C.DEVICE = 'cuda:0'
_C.EPOCHS = 50
_C.BATCH_SIZE = 64

_C.INPUT_SIZE = 224
_C.WORKERS = 8

_C.pretrain_path = None
_C.BA = 0  # best acc from pretrained model
###########################
# Lambda
###########################
_C.LAMBDA = CN()
_C.LAMBDA.cal = 1e-1  # lambda self-calibration loss
_C.LAMBDA.distill = 1e-3  # lambda distillation loss
_C.LAMBDA.pred = (0.01, 0.01)  # lambda distillation loss

###########################
# Model
###########################
_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.CLIP = 'ViT-B/16'
_C.MODEL.RN = 'RN101'

###########################
# Optimizer
###########################

_C.OPT = CN()
_C.OPT.lr = 1e-3
_C.OPT.lr_dense = 1e-4
# _C.OPT.weight_decay = 0.05
# _C.OPT.layer_decay = 0.65
# _C.OPT.fix_layer = 10
# _C.OPT.warn_up = 10

###########################
# Dataset
###########################

_C.DATASET = CN()
_C.DATASET.NAME = ''
# dataset file path
_C.DATASET.PATH = ''
# dataset info/split pkl file path
_C.DATASET.PKL = ''


def get_cfg_defaults():
    return _C.clone()