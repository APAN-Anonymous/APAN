
from typing import List, Tuple, Union

import numpy
import torch
from yacs.config import CfgNode
import torchvision.models.resnet as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from clip.model import QuickGELU, LayerNorm
from utils import *


class InteractAttention(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()

        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(y_dim, x_dim)))
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(y_dim, x_dim)))
        self.W_3 = nn.Parameter(nn.init.zeros_(torch.empty(y_dim, x_dim)))

    def forward(self, x, y, att2cls, vec_bias=None):
        ############### dense attention #################
        #####
        # einstein sum notation
        # b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes
        # i: number of attribute \ h : hidden attention dim
        #####

        x = F.normalize(x, dim=1)
        # Compute attribute score on each image region
        S = torch.einsum('iv,vf,bfr->bir', y, self.W_1, x)
        # compute Dense Attention
        A = torch.einsum('iv,vf,bfr->bir', y, self.W_2, x)

        ## Ablation setting - with no attribute attention
        # R = x.size(2)
        # B = x.size(0)
        # dim_att = y.size(0)
        # A_b = x.new_full((B, dim_att, R), 1 / R)
        # A = A_b

        A = F.softmax(A, dim=-1)  # compute an attention map for each attribute

        F_p = torch.einsum('bir,bfr->bif', A, x)  # compute attribute-based features h_i^a

        S_p = torch.einsum('bir,bir->bi', A, S)  # compute attribute scores from attribute attention maps

        # compute Attention over Attribute
        A_p = torch.einsum('iv,vf,bif->bi', y, self.W_3, F_p)
        # attribute in the image i
        A_p = torch.sigmoid(A_p)

        # compute the final prediction as the product of semantic scores, attribute scores, and attention over
        # attribute scores
        S_pp = torch.einsum('ki,bi,bi->bik', att2cls.T, A_p, S_p)
        S_pp = torch.sum(S_pp, axis=1)  # [bk] <== [bik]

        #bais mask
        if vec_bias is not None:
            S_pp = S_pp + vec_bias

        return S_pp


class PureCLIP(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.model = clip_model
        self.dtype = self.model.dtype

    def forward(self, image: torch.Tensor, classnames: torch.LongTensor):
        logits_per_image, logits_per_text = self.model(image, classnames)
        return logits_per_image

    def forward1(self, image: torch.Tensor, classnames: torch.LongTensor):
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(classnames)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity

    def prediction(self, x, classnames):
        global_logits = self.forward(x, classnames)
        return global_logits


class APAN(nn.Module):
    def __init__(self, clip_model: nn.Module, rn_model: nn.Module, dataloader: UNIDataloader, attributes: torch.Tensor):
        super().__init__()
        # read info from dataset
        self.seenclasses = dataloader.seenclasses
        self.unseenclasses = dataloader.unseenclasses
        self.nclass = len(dataloader.classnames)

        # clip visual
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

        # resnet
        self.resnet = rn_model

        self.attributes = nn.Parameter(attributes.clone().detach(), requires_grad=True)

        # dense attention
        mask_bias = torch.ones((1, self.nclass))
        mask_bias[:, self.seenclasses] *= -1
        self.vec_bias = nn.Parameter(mask_bias, requires_grad=False)

        self.dense_attn_clip = InteractAttention(512, 512)
        self.dense_attn_resnet = InteractAttention(2048, 512)

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward_clip_visual(self, x):
        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def forward(self, x: torch.Tensor, classnames: torch.Tensor, att2cls: torch.Tensor):
        with torch.no_grad():
            x_clip = self.forward_clip_visual(x)
            x_clip = x_clip @ self.proj
            x_resnet = self.resnet(x)

        # global stream
        cls_token = x_clip[:, 0, :]
        global_logits = cos_sim(cls_token, classnames)

        # local stream clip
        local_tokens = x_clip[:, 1:, :]
        clip_dense = self.dense_attn_clip(local_tokens.permute(0, 2, 1), self.attributes, att2cls,
                                          self.vec_bias)  # brf -> bfr
        # local stream res101
        shape = x_resnet.shape
        x_resnet = x_resnet.reshape(shape[0], shape[1], shape[2] * shape[3])
        rn_dense = self.dense_attn_resnet(x_resnet, self.attributes, att2cls, self.vec_bias)

        return global_logits, clip_dense, rn_dense

    def compute_loss_Self_Calibrate(self, S_pp):
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, self.unseenclasses]
        assert Prob_unseen.size(1) == len(self.unseenclasses)
        mass_unseen = torch.sum(Prob_unseen, dim=1)

        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_align_loss(self, embedding1, embedding2):
        """
        prediction alignment loss
        """
        S_pp1, S_pp2 = embedding1, embedding2
        wt = (S_pp1 - S_pp2).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
        loss = wt * (S_pp1 - S_pp2).abs()
        loss = (loss.sum() / loss.size(0))

        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        p_output = F.softmax(S_pp1, dim=1)
        q_output = F.softmax(S_pp2, dim=1)
        log_mean_output = ((p_output + q_output) / 2).log()
        loss += (KLDivLoss(log_mean_output, q_output) + KLDivLoss(log_mean_output, p_output)) / 2

        return loss

    def compute_aug_cross_entropy(self, S_pp, labels):
        if self.vec_bias is not None:
            S_pp = S_pp - self.vec_bias  # remove the margin +1/-1 from prediction scores
        loss = F.cross_entropy(S_pp, labels)
        return loss

    def prediction(self, x, classnames, att2cls, lambda_clip=0.01, lambda_rn=None, out: str = None):
        if lambda_rn is None:
            lambda_rn = lambda_clip
        global_logits, clip_local, rn_local = self.forward(x, classnames, att2cls)

        return global_logits + lambda_clip * clip_local + lambda_rn * rn_local

    def save(self, file_path: str):
        mkdir(os.path.dirname(file_path))
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str, device: Union[str, torch.device] = 'cuda'):
        assert os.path.isfile(file_path)
        weight = torch.load(file_path, map_location=device)
        self.load_state_dict(weight)


def cos_sim(x, y):
    assert x.shape[-1] == y.shape[-1], \
        f'the input should have the same dimension, but x:{x.shape[-1]},  y:{y.shape[-1]}'
    y = F.normalize(y)
    return x @ y.T

