from pathlib import Path
import multiprocessing

import pandas as pd
import pickle
import numpy as np
import tqdm
import torch

from src.dataset import SeverStalDS, dev_transform, collate_fn
from src.metric import calc_dice


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_data(path):
    train = pd.read_csv(path)
    train['ImageId'], train['ClassId'] = zip(*train.ImageId_ClassId.str.split('_'))
    train['ClassId'] = train['ClassId'].astype(int)

    return list(train.groupby('ImageId'))


def calc_dice_for_all(dice_thresh):
    n_classes = 4
    dices = np.zeros((len(preds), n_classes))
    for i, (p, t) in enumerate(zip(preds, targets)):
        for c in range(n_classes):
            dices[i, c] = calc_dice(p[c], t[c], thresh=dice_thresh)

    return dice_thresh, dices


def main():
    global preds, targets
    PATH_TO_DATA = Path('/home/kbrodt/kaggle/data/severstal-steel-defect-detection/')
    train_gps = get_data(PATH_TO_DATA / 'train.csv.zip')
    root = Path('./brave5')

    train_ds = SeverStalDS(train_gps, root=root / 'pred_masks_tta', transform=dev_transform, preds=True)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=32,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=collate_fn)

    preds, targets = [], []
    with tqdm.tqdm(train_loader) as pbar:
        for x, y in pbar:
            preds.extend(x.numpy())
            targets.extend(y.numpy())

    cls_preds = np.array([np.load(root / 'pred_clss_tta' / (p + '.npy')) for p, _ in tqdm.tqdm(train_gps)])
    cls_trgs = np.array([(t.sum((-1, -2)) > 0) for t in tqdm.tqdm(targets)])
    assert len(preds) == len(targets) == len(cls_preds) == len(cls_trgs)

    threshs = np.linspace(0.05, 0.95, 19)

    with multiprocessing.Pool(len(threshs)) as p:
        with tqdm.tqdm(threshs) as pbar:
            res = list(p.imap_unordered(func=calc_dice_for_all, iterable=pbar))
    res = {th: r for th, r in res}

    cls_preds_bin = {}
    threshs_cls = np.linspace(0.00, 1, 21)
    with tqdm.tqdm(threshs_cls) as pbar:
        for th in pbar:
            cls_preds_bin[th] = cls_preds <= th

    def step(cls_thresh, dice_thresh):
        dice = np.zeros(4)
        for c_p, d, t in zip(cls_preds, res[dice_thresh], cls_trgs):
            for c in range(4):
                if c_p[c] <= cls_thresh:
                    if t[c]:
                        dice[c] += 0
                    else:
                        dice[c] += 1
                    continue

                dice[c] += d[c]

        return dice / len(cls_preds)

    results = {}
    with tqdm.tqdm(threshs_cls) as pbar:
        for th_cls in pbar:
            for th_dice in threshs:
                asd = step(th_cls, th_dice)
                for c in range(4):
                    results[th_cls, th_dice, c] = asd[c]

    d = sum([(list(filter(lambda x: x[0][-1] == c, sorted(results.items(), key=lambda x: -x[1]))))[0][-1]
             for c in range(4)])/4
    print(d)

    for c in range(4):
        print(list(filter(lambda x: x[0][-1] == c, sorted(results.items(), key=lambda x: -x[1])))[0])


if __name__ == '__main__':
    main()
