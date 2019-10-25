import argparse
from pathlib import Path
import random
import os

import apex
import numpy as np
import torch
import tqdm

from src.dataset import SeverStalDS, dev_transform, collate_fn
from src.metric import Dice
from src.utils import get_data_groups
import segmentation_models as smp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/data/severstal-steel-defect-detection',
                        help='Path to data')
    parser.add_argument('--load', type=str, required=True,
                        help='Load model')
    parser.add_argument('--save', type=str, default='',
                        help='Save predictions')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def epoch_step(loader, desc, model, metric):
    model.eval()

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_targets, loc_preds = [], []
    loc_preds_cls = []
    
    
    for x, y in loader:
        x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()
        
        masks, clsss = [], []
        logits = model(x)
        if args.cls:
            logits, cls = logits
            clsss.append(torch.sigmoid(cls).cpu().numpy())
        masks.append(torch.sigmoid(logits).cpu().numpy())
            
        logits = model(torch.flip(x, dims=[-1]))
        if args.cls:
            logits, cls = logits
            clsss.append(torch.sigmoid(cls).cpu().numpy())
        masks.append(torch.flip(torch.sigmoid(logits), dims=[-1]).cpu().numpy())
            
        logits = model(torch.flip(x, dims=[-2]))
        if args.cls:
            logits, cls = logits
            clsss.append(torch.sigmoid(cls).cpu().numpy())
        masks.append(torch.flip(torch.sigmoid(logits), dims=[-2]).cpu().numpy())
            
        logits = model(torch.flip(x, dims=[-1, -2]))
        if args.cls:
            logits, cls = logits
            clsss.append(torch.sigmoid(cls).cpu().numpy())
        masks.append(torch.flip(torch.sigmoid(logits), dims=[-1, -2]).cpu().numpy())

        trg = y.cpu().numpy()
        loc_targets.extend(trg)
        preds = np.mean(masks, 0)
        loc_preds.extend(preds)
        metric.update(preds, trg)
        
        if args.cls:
            loc_preds_cls.extend(np.mean(clsss, 0))

        torch.cuda.synchronize()

        if args.local_rank == 0:
            pbar.set_postfix(**{
                'dice': f'{metric.evaluate():.4}',
            })
            pbar.update()

    pbar.close()

    return loc_targets, loc_preds, loc_preds_cls

    
def main():
    global args
    
    args = parse_args()

    torch.backends.cudnn.benchmark = True

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
    
    to_save = Path(args.save)
    path_to_load = Path(args.load)
    if path_to_load.is_file():
        print(f"=> Loading checkpoint '{path_to_load}'")
        checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
        print(f"=> Loaded checkpoint '{path_to_load}'")
    else:
        raise
    args = checkpoint['args']
    print(args)

    n_classes = args.n_classes
    if args.cls:
        print('With classification')
    else:
        print('Without classification')

    model = smp.Unet(encoder_name=args.encoder,
                     encoder_weights='imagenet',
                     classes=n_classes,
                     activation='sigmoid',
                     n_classes=n_classes if args.cls else None)
    
    if args.sync_bn:
        print('using apex synced BN')
        model = apex.parallel.convert_syncbn_model(model)
        
    model.cuda()
     
    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.fp16:
        model = apex.amp.initialize(model,
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
        
    # work_dir = Path(args.work_dir)
    work_dir = path_to_load.parent
    
    model.load_state_dict(checkpoint['state_dict'])
    
    path_to_data = Path(args.data)
    _, dev_gps = get_data_groups(path_to_data / 'train.csv.zip', args)
    
    dev_ds = SeverStalDS(dev_gps, root=path_to_data / 'train', transform=dev_transform)
    dev_sampler = None
    if args.distributed:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)
        
    batch_size = args.batch_size
    dev_loader = torch.utils.data.DataLoader(dev_ds,
                                             batch_size=min(batch_size, 16),
                                             shuffle=False,
                                             sampler=dev_sampler,
                                             num_workers=4,
                                             collate_fn=collate_fn,
                                             pin_memory=True)

    metric = Dice(n_classes=n_classes, thresh=0.5)
    
    x = torch.rand(2, 3, 256, 256).cuda()
    model = model.eval()
    model.encoder.set_swish(memory_efficient=False)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)

    traced_model.save(str(work_dir / f'model_{path_to_load.stem}.pt'))
    del traced_model
    del model

    model = torch.jit.load(str(work_dir / f'model_{path_to_load.stem}.pt')).cuda().eval()

        
    with torch.no_grad():
        metric.clean()
        trgs, preds, preds_cls = epoch_step(dev_loader, f'[ Validating dev.. ]',
                                            model=model,
                                            metric=metric)
        print(f'dice dev {metric.evaluate()}')
    if str(to_save) == '':
        return
    to_save1 = to_save / 'pred_masks_tta'
    if not to_save1.exists():
        to_save1.mkdir(parents=True)
    to_save2 = to_save / 'pred_clss_tta'
    if not to_save2.exists():
        to_save2.mkdir(parents=True)
    with tqdm.tqdm(zip(dev_gps, preds, preds_cls), total=len(preds)) as pbar:
        for (fname, _), p1, p2 in pbar:
            np.save(to_save1 / fname, p1)
            np.save(to_save2 / fname, p2)


if __name__ == '__main__':
    main()
