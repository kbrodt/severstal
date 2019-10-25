import argparse
from pathlib import Path
import random
import os

import apex
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from src.dataset import SeverStalDS, train_transform, dev_transform, collate_fn
from src.lovasz_losses import symmetric_lovasz
from src.metric import Dice
from src.utils import get_data_groups
import segmentation_models as sm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/kbrodt/kaggle/data/severstal-steel-defect-detection',
                        help='Path to data')
    parser.add_argument('--load', default='', type=str,
                        help='Load model (default: none)')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0')
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--n-classes', type=int, default=4)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    
    parser.add_argument('--workers', '-j', type=int, default=4, required=False)

    parser.add_argument('--epochs', '-e', type=int, default=130)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='Batch size per process (default: 8)')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', '--learning-rate', type=float, default=5e-4,
                        metavar='LR',
                        help='Initial learning rate. Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=5)
    
    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true')
    
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enabling apex sync BN.')
    
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def epoch_step(loader, desc, model, criterion, metric, opt=None, batch_accum=1):
    is_train = opt is not None
    if is_train:
        model.train()
    else:
        model.eval()

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_loss = n = 0
    loc_accum = 1
    for x, y in loader:
        x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(args.gpu, non_blocking=True).float()
        logits = model(x)
        loss = criterion(logits, y) / batch_accum
        if is_train:
            if args.fp16:
                with apex.amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if loc_accum == batch_accum:
                opt.step()
                opt.zero_grad()
                loc_accum = 1
            else:
                loc_accum += 1
            if args.cls:
                logits, _ = logits
            logits = logits.detach()
        elif args.cls:  # inference
            logits, _ = logits

        bs = x.size(0)
        loc_loss += loss.item() * bs * batch_accum
        n += bs

        metric.update(torch.sigmoid(logits).cpu().numpy(), y.cpu().numpy())

        torch.cuda.synchronize()

        if args.local_rank == 0:
            pbar.set_postfix(**{
                'loss': loc_loss / n,
                'dice': f'{metric.evaluate():.4}',
            })
            pbar.update()

    if is_train and loc_accum != batch_accum:
        opt.step()
        opt.zero_grad()
    pbar.close()

    return loc_loss / n


def plot_hist(history, path):
    history_len = len(history)
    n_rows = history_len//2 + 1
    n_cols = 2
    plt.figure(figsize=(12, 4*n_rows))
    for i, (m, vs) in enumerate(history.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        for k, v in vs.items():
            plt.plot(v, label=f'{k} {v[-1]:.4}')

        plt.xlabel('#epoch')
        plt.ylabel(f'{m}')
        plt.legend()
        plt.grid(ls='--')

    plt.tight_layout()
    plt.savefig(path / 'evolution.png')
    plt.close()


def main():
    global args

    args = parse_args()
    print(args)
    
    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)
        
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'
    
    # create model
    n_classes = args.n_classes
    if args.cls:
        print('With classification')
    else:
        print('Without classification')

    model = sm.Unet(encoder_name=args.encoder,
                    encoder_weights='imagenet',
                    classes=n_classes,
                    activation='sigmoid',
                    n_classes=n_classes if args.cls else None)

    if args.sync_bn:
        print('using apex synced BN')
        model = apex.parallel.convert_syncbn_model(model)
        
    model.cuda()
    
    # Scale learning rate based on global batch size
    print(f'lr={args.lr}, opt={args.opt}')
    if args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(), args.lr)
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
                             )
    else:
        raise
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=1e-5, last_epoch=-1)

    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.fp16:
        model, opt = apex.amp.initialize(model, opt,
                                         opt_level=args.opt_level,
                                         keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                         loss_scale=args.loss_scale
                                        )
    
    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        
    if args.cls:
        def BCEBCE(logits, target):
            prediction_seg, prediction_cls = logits
            y_cls = (target.sum([2, 3]) > 0).float()

            return sm.utils.losses.BCEDiceFocalLoss()(prediction_seg, target) + nn.BCEWithLogitsLoss()(prediction_cls, y_cls)

        def symmetric_lovasz_fn(logits, target):
            prediction_seg, prediction_cls = logits
            y_cls = (target.sum([2, 3]) > 0).float()

            return symmetric_lovasz(prediction_seg, target) + nn.BCEWithLogitsLoss()(prediction_cls, y_cls)
    else:
        BCEBCE = sm.utils.losses.BCEDiceFocalLoss()
        symmetric_lovasz_fn = symmetric_lovasz

    criterion = BCEBCE
    
    history = {
        k: {k_: [] for k_ in ['train', 'dev']}
        for k in ['loss', 'dice']
    }
    best_dice = 0  # 0.946
    
    base_name = f'{args.encoder}_b{args.batch_size}_{args.opt}_lr{args.lr}_c{int(args.cls)}_fold{args.fold}'
    work_dir = Path(base_name)
    if args.local_rank == 0 and not work_dir.exists():
        work_dir.mkdir(parents=True)
    
    # Optionally load model from a checkpoint
    if args.load:
        def _load():
            path_to_load = Path(args.load)
            if path_to_load.is_file():
                print(f"=> loading model '{path_to_load}'")
                checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
                model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded model '{path_to_load}'")
            else:
                print(f"=> no model found at '{path_to_load}'")
        _load()
    
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def _resume():
            nonlocal history, best_dice
            path_to_resume = Path(args.resume)
            if path_to_resume.is_file():
                print(f"=> loading resume checkpoint '{path_to_resume}'")
                checkpoint = torch.load(path_to_resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch'] + 1
                history = checkpoint['history']
                best_dice = checkpoint['best_dice']
                model.load_state_dict(checkpoint['state_dict'])
                opt.load_state_dict(checkpoint['opt_state_dict'])
                print(f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.resume}'")
        _resume()
    
    path_to_data = Path(args.data)
    train_gps, dev_gps = get_data_groups(path_to_data / 'train.csv.zip', args)
    
    train_ds = SeverStalDS(train_gps, root=path_to_data / 'train', transform=train_transform)
    dev_ds = SeverStalDS(dev_gps, root=path_to_data / 'train', transform=dev_transform)

    train_sampler = None
    dev_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)
        
    batch_size = args.batch_size
    num_workers = args.workers
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               shuffle=train_sampler is None,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=True)

    dev_loader = torch.utils.data.DataLoader(dev_ds,
                                             batch_size=min(batch_size, 16),
                                             shuffle=False,
                                             sampler=dev_sampler,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn,
                                             pin_memory=True)

    metric = Dice(n_classes=n_classes, thresh=0.5)

    saver = lambda path: torch.save({
        'epoch':  epoch,
        'best_dice': best_dice,
        'history': history,
        'state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'args': args,
    }, path)
    to_adjust = True
    lr_reduce_eps = 100
    to_lovas_eps = 60
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= to_lovas_eps:
            criterion = symmetric_lovasz_fn
        if to_adjust and epoch >= lr_reduce_eps:
            print('lr reduced')
            sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', min_lr=5e-6,
                                                                  patience=args.patience, factor=0.5, verbose=True)
            to_adjust = False

        metric.clean()
        loss = epoch_step(train_loader, f'[ Training {epoch}/{args.epochs}.. ]',
                          model=model, criterion=criterion, metric=metric, opt=opt, batch_accum=1)
        history['loss']['train'].append(loss)
        history['dice']['train'].append(metric.evaluate())

        with torch.no_grad():
            metric.clean()
            loss = epoch_step(dev_loader, f'[ Validating {epoch}/{args.epochs}.. ]',
                              model=model, criterion=criterion, metric=metric, opt=None)
            history['loss']['dev'].append(loss)
            history['dice']['dev'].append(metric.evaluate())
        if epoch < lr_reduce_eps:
            sheduler.step()
        else:
            sheduler.step(history['dice']['dev'][-1])

        if args.local_rank == 0:
            saver(work_dir / 'last.pth')
            if history['dice']['dev'][-1] > best_dice:
                best_dice = history['dice']['dev'][-1]
                saver(work_dir / 'best.pth')

            plot_hist(history, work_dir)


if __name__ == '__main__':
    main()
