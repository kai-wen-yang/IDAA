import os
import torch
import torchvision
import argparse
import sys
from torch.autograd import Variable
import numpy as np
import wandb
import torchvision.transforms as transforms
from model import load_model, save_model
from modules import NT_Xent
from modules.transformations import TransformsSimCLR
from utils import mask_correlated_samples
from eval_knn import kNN
sys.path.append('..')
from set import *
from vae import *
from apex import amp


parser = argparse.ArgumentParser(description=' Seen Testing Category Training')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--dim', default=512, type=int, help='CNN_embed_dim')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=300, type=int, help='epochs')
parser.add_argument('--save_epochs', default=100, type=int, help='save epochs')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--projection_dim', default=64, type=int, help='projection_dim')
parser.add_argument('--optimizer', default="Adam", type=str, help="optimizer")
parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='weight_decay')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature')
parser.add_argument('--model_path', default='log/', type=str,
                    help='model save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str,
                    help='model save path')

parser.add_argument('--dataset', default='cifar10',
                    help='[cifar10, cifar100]')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='batch norm momentum for advprop')
parser.add_argument('--alpha', default=1.0, type=float, help='weight for contrastive loss with adversarial example')
parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
parser.add_argument('--vae_path',
                    default='../results/vae_dim512_kl0.1_simclr/model_epoch92.pth',
                    type=str, help='vae_path')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt_level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
args = parser.parse_args()
print(args)
set_random_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def gen_adv(model, vae, x_i, criterion, optimizer):
    x_i = x_i.detach()
    h_i, z_i = model(x_i, adv=True)

    with torch.no_grad():
        z, gx, _, _ = vae(x_i)
    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, True)
    x_j_adv = adv_gx + (x_i - gx).detach()
    h_j_adv, z_j_adv = model(x_j_adv, adv=True)
    tmp_loss = criterion(z_i, z_j_adv)
    if args.amp:
        with amp.scale_loss(tmp_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle.data = variable_bottle.data + args.eps * sign_grad
        adv_gx = vae(variable_bottle, True)
        x_j_adv = adv_gx + (x_i - gx).detach()
    x_j_adv.requires_grad = False
    x_j_adv.detach()
    return x_j_adv, gx


def train(args, epoch, train_loader, model, vae, criterion, optimizer):
    model.train()
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        if args.adv:
            x_j_adv, gx = gen_adv(model, vae,  x_i, criterion, optimizer)

        optimizer.zero_grad()
        h_j, z_j = model(x_j)
        loss_og = criterion(z_i, z_j)
        if args.adv:
            _, z_j_adv = model(x_j_adv, adv=True)
            loss_adv = criterion(z_i, z_j_adv)
            loss = loss_og + args.alpha * loss_adv
        else:
            loss = loss_og
            loss_adv = loss_og
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"[Epoch]: {epoch} [{step}/{len(train_loader)}]\t Loss: {loss.item():.3f} Loss_og: {loss_og.item():.3f} Loss_adv: {loss_adv.item():.3f}")

        loss_epoch += loss.item()
        args.global_step += 1

        if args.debug:
            break
        if step % 10 == 0:
            wandb.log({'loss_og': loss_og.item(),
                       'loss_adv': loss_adv.item(),
                       'lr': optimizer.param_groups[0]['lr']})
        if args.global_step % 1000 == 0:
            if args.adv:
                reconst_images(x_i, gx, x_j_adv)
    return loss_epoch


def main():
    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_sampler = None
    if args.dataset == "cifar10":
        root = "../../data"
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimCLR()
        )
        data = 'non_imagenet'
        transform_test = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
        vae = CVAE_cifar_withbn(128, args.dim)
    elif args.dataset == "cifar100":
        root = "../../data"
        train_dataset = torchvision.datasets.CIFAR100(
            root, download=True, transform=TransformsSimCLR()
        )
        data = 'non_imagenet'
        transform_test = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
        vae = CVAE_cifar_withbn(128, args.dim)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100, shuffle=False, num_workers=4)

    ndata = train_dataset.__len__()
    log_dir = "log/" + args.dataset + '_log/'

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    if args.adv:
        suffix = suffix + '_alpha_{}_adv_eps_{}'.format(args.alpha, args.eps)
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag=True,
                                                 bn_adv_momentum=args.bn_adv_momentum, data=data)
    else:
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag=False,
                                                 bn_adv_momentum=args.bn_adv_momentum, data=data)

    vae.load_state_dict(torch.load(args.vae_path))
    vae.to(args.device)
    vae.eval()
    if args.amp:
        [model, vae], optimizer = amp.initialize(
            [model, vae], optimizer, opt_level=args.opt_level)

    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim)
    suffix = suffix + '_bn_adv_momentum_{}_seed_{}'.format(args.bn_adv_momentum, args.seed)
    wandb.init(config=args, name=suffix.replace("_log/", ''))

    test_log_file = open(log_dir + suffix + '.txt', "w")

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_dir = args.model_dir + args.dataset + '/'
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    mask = mask_correlated_samples(args)
    criterion = NT_Xent(args.batch_size, args.temperature, mask, args.device)

    args.global_step = 0
    args.current_epoch = 0
    best_acc = 0
    for epoch in range(0, args.epochs):
        loss_epoch = train(args, epoch, train_loader, model, vae, criterion, optimizer)
        model.eval()
        if epoch > 10:
            scheduler.step()
        print('epoch: {}% \t (loss: {}%)'.format(epoch, loss_epoch / len(train_loader)), file=test_log_file)
        print('----------Evaluation---------')
        start = time.time()
        acc = kNN(epoch, model, train_loader, testloader, 200, args.temperature, ndata, low_dim=args.projection_dim)
        print("Evaluation Time: '{}'s".format(time.time() - start))

        if acc >= best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(state, args.model_dir + suffix + '_best.t')
            best_acc = acc
        print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc))
        print('[Epoch]: {}'.format(epoch), file=test_log_file)
        print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc), file=test_log_file)
        wandb.log({'acc': acc})
        test_log_file.flush()

        args.current_epoch += 1
        if args.debug:
            break
        if epoch % 50 == 0:
            save_model(args.model_dir + suffix, model, optimizer, epoch)

    save_model(args.model_dir + suffix, model, optimizer, args.epochs)


def reconst_images(x_i, gx, x_j_adv):
    grid_X = torchvision.utils.make_grid(x_i[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"X.jpg": [wandb.Image(grid_X)]}, commit=False)
    grid_GX = torchvision.utils.make_grid(gx[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"GX.jpg": [wandb.Image(grid_GX)]}, commit=False)
    grid_RX = torchvision.utils.make_grid((x_i[32:96] - gx[32:96]).data, nrow=8, padding=2, normalize=True)
    wandb.log({"RX.jpg": [wandb.Image(grid_RX)]}, commit=False)
    grid_AdvX = torchvision.utils.make_grid(x_j_adv[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"AdvX.jpg": [wandb.Image(grid_AdvX)]}, commit=False)
    grid_delta = torchvision.utils.make_grid((x_j_adv - x_i)[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"Delta.jpg": [wandb.Image(grid_delta)]}, commit=False)
    wandb.log({'l2_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).norm(dim=1)),
               'linf_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).abs().max(dim=1)[0])
               }, commit=False)


if __name__ == "__main__":
    main()
