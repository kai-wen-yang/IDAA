# IDAA

Official implementation:
- Identity-Disentangled Adversarial Augmentation for Self-Supervised Learning, ICML 2022. ([Paper](https://proceedings.mlr.press/v162/yang22s/yang22s.pdf))


<div align="center">
  <img src="IDAA.png" width="1000px" />
  <p>Architecture and pipeline of Identity-Disentangled Adversarial Augmentation (IDAA)</p>
</div>

For questions, you can contact (kwyang@mail.ustc.edu.cn).

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

SimCLR training and evaluation:
```
python main.py --seed 1 --gpu 0  --dataset cifar10 --resnet resnet18;
python eval_lr.py --seed 1 --gpu 0 --dataset cifar10 --resnet resnet18
```
SimCLR+IDAA training and evaluation:
```
python main.py --adv --eps 0.1 --seed 1 --gpu 0 --dataset cifar10 --dim 512 --vae_path ../results/vae_cifar10_dim512_kl0.1_simclr/model_epoch292.pth --resnet resnet18;
python eval_lr.py --adv --eps 0.1 --seed 1 --gpu 0 --dataset cifar10 --dim 512 --resnet resnet18
```

## References
We borrow some code from https://github.com/chihhuiho/CLAE.


## Citation

If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{yang2022identity,
  title={Identity-Disentangled Adversarial Augmentation for Self-supervised Learning},
  author={Yang, Kaiwen and Zhou, Tianyi and Tian, Xinmei and Tao, Dacheng},
  booktitle={International Conference on Machine Learning},
  pages={25364--25381},
  year={2022},
  organization={PMLR}
}
```
