
import argparse
import time

import config
from model import APAN
from utils import *



def test(cfg):
    device = torch.device(cfg.DEVICE)

    setup_seed(cfg.SEED)

    # build model
    clip_model, preprocess = load_clip(cfg.MODEL.CLIP, device=device)
    convert_models_to_fp32(clip_model)
    dataloader = UNIDataloader(cfg, preprocess)
    rn_model = load_resnet(cfg.MODEL.RN)

    # prompt setting
    classnames = clip.tokenize([f"a photo of a {name}" for name in dataloader.classnames]).to(device)
    attributes = clip.tokenize([f"{att}" for att in dataloader.attributes]).to(device)

    # to tensor
    att2cls = dataloader.att2cls.to(device)
    with torch.no_grad():
        classnames = clip_model.encode_text(classnames)
        attributes = clip_model.encode_text(attributes)

    model = APAN(clip_model, rn_model, dataloader, attributes)
    if cfg.pretrain_path:
        model.load(cfg.pretrain_path, device=device)
    convert_models_to_fp32(model)
    model.to(device)

    acc_zs, acc_novel, acc_seen, H = benchmark(cfg, dataloader, model, [classnames, att2cls, *cfg.LAMBDA.pred])

    if cfg.TASK == 'CZSL':
        print(f'CZSL Acc result: {acc_zs * 100:.3f}\t{acc_novel * 100:.3f}\t{acc_seen * 100:.3f}\t{H * 100:.3f}\t')
    elif cfg.TASK == 'GZSL':
        print(f'GZSL result: U = {acc_novel * 100:.3f} | S = {acc_seen * 100:.3f} | H = {H * 100:.3f}\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', "--dataset", type=str, default='CUB')
    parser.add_argument('-t', "--task", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help='0, 1 refer to cuda:0, cuda:1; c for cpu')
    parser.add_argument('-p', "--pretrain", type=str, default=None, help='pretrained model path')

    args = parser.parse_args()
    if args.pretrain:
        info = args.pretrain.split('/')
        args.dataset = info[-3]
        cfg = config.get_dataset_cfg(args.dataset)
        cfg.pretrain_path = args.pretrain
        cfg.MODEL.NAME = info[-4]
        cfg.TASK = info[-2]
        cfg.BA = info[-1].split('_')[0]  # best acc
    else:
        cfg = config.get_dataset_cfg(args.dataset)

    if args.device:
        if args.device.isdigit():
            cfg.DEVICE = f'cuda:{args.device}'
        elif args.device == 'c':
            cfg.DEVICE = 'cpu'
        else:
            raise ValueError('device should be a number or c')

    print(cfg)

    test(cfg)
