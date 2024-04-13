
import clip
import torchvision.models.resnet as models
import torch.nn as nn
from .dataset import UNIDataloader
from .orig_eva import evaluation, benchmark
from .misc import *

_MODELS = {
    "RN50": "backbone_pretrained/CLIP/RN50.pt",
    "RN101": "backbone_pretrained/CLIP/RN101.pt",
    "RN50x4": "backbone_pretrained/CLIP/RN50x4.pt",
    "RN50x16": "backbone_pretrained/CLIP/RN50x16.pt",
    "RN50x64": "backbone_pretrained/CLIP/RN50x64.pt",
    "ViT-B/32": "backbone_pretrained/CLIP/ViT-B-32.pt",
    "ViT-B/16": "backbone_pretrained/CLIP/ViT-B-16.pt",
    "ViT-L/14": "backbone_pretrained/CLIP/ViT-L-14.pt",
    "ViT-L/14@336px": "backbone_pretrained/CLIP/ViT-L-14-336px.pt",
}

_RESNETS = {
    "RN50": "backbone_pretrained/ResNet/resnet50-0676ba61.pth",
    "RN101": "backbone_pretrained/ResNet/resnet101-63fe2227.pth",
}


def load_clip(backbone: str, **kwargs):
    assert backbone in clip.available_models(), \
        f'backbone must be one of them: {clip.available_models()}, but get {backbone} '
    model, preprocess = clip.load(_MODELS[backbone], **kwargs)
    return model, preprocess


def load_resnet(backbone: str, feature_map: bool = True, eval: bool = True):
    assert backbone in _RESNETS.keys(), \
        f'backbone must be one of them: {_RESNETS.keys()}, but get {backbone} '
    model = models.resnet101(pretrained=False)
    model.load_state_dict(torch.load(_RESNETS[backbone]))
    if feature_map: model = nn.Sequential(*list(model.children())[:-2])
    if eval: model.eval()
    return model
