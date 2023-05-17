# APAN
 
> A implement code for 
> 
> *ZSL Meets CLIP: An Attribute Prompt Alignment Network for Zero-Shot Learning*

## Backbones
pretrained backbone file show placed here:
```
├─backbone_pretrained
   ├─CLIP
   │  └─ViT-B-16.pt
   └─ResNet
      └─resnet101-63fe2227.pth
```

## Datasets prepare
dataset files should be placed like below:
```
├─data
   ├─AWA2
   │  ├─Animals_with_Attributes2
   │  └─AWA2_full.pkl
   ├─CUB
   │  ├─CUB_200_2011
   │  └─CUB_full.pkl
   └─SUN
      ├─images
      └─SUN_full.pkl
```

## APAN pretrain model
pretrained model file show placed here:
```
├─pretrined
   ├─AWA2
   │  ├─CZSL
   │  │  └─xxx.pth
   │  └─GZSL
   │     └─xxx.pth
   ├─CUB
   │  ├─CZSL
   │  │  └─xxx.pth
   │  └─GZSL
   │     └─xxx.pth
   └─SUN
      ├─CZSL
      │  └─xxx.pth
      └─GZSL
         └─xxx.pth
```