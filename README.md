# IDAA

Official implementation:
- Identity-Disentangled Adversarial Augmentation for Self-supervised Learning, ICML 2022. ([Paper](https://proceedings.mlr.press/v162/yang22s/yang22s.pdf))

<div align="center">
  <img src="IDAA.png" width="1000px" />
  <p>Architecture and pipeline of Identity-Disentangled Adversarial Augmentation (IDAA)</p>
</div>

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Wandb](https://wandb.ai/site)
4. [Torchvision](https://pytorch.org/vision/stable/index.html)
5. [Apex(optional)](https://github.com/NVIDIA/apex)

## Pretrain a VAE

```
python train_vae.py --dim 512 --kl 0.1 --save_dir ./results/vae_cifar10_dim512_kl0.1_simclr --mode simclr --dataset cifar10
```

## Apply IDAA to SimCLR
```
cd SimCLR
```

Train a original SimCLR and evaluate it:
```
python main.py --eps 0.1  --dataset CIFAR10 --resnet resnet18;
python eval_lr.py --eps 0.1 --dataset CIFAR10 --resnet resnet18
```

```
python main.py --adv --eps 0.1  --dataset CIFAR10 --dim 512 --vae_path ../results/vae_cifar10_dim512_kl0.1_simclr/model_epoch292.pth --resnet resnet18;
python eval_lr.py --adv --eps 0.1 --dataset CIFAR10 --dim 512 --resnet resnet18
```
