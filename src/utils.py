import pickle

import jpeg4py
import numpy as np
import pandas as pd
import pycocotools.mask as mutils
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_data_groups(path, args):
    train = pd.read_csv(path)
    train['ImageId'], train['ClassId'] = zip(*train.ImageId_ClassId.str.split('_'))
    train['ClassId'] = train['ClassId'].astype(int)
    train_gps = list(train.groupby('ImageId'))
    y = np.array([[0 if gp.iloc[i].EncodedPixels != gp.iloc[i].EncodedPixels else 1 for i in range(4)]
                  for _, gp in train_gps])
    mskf = MultilabelStratifiedKFold(n_splits=args.n_folds, random_state=args.seed)
    for i, (train_index, dev_index) in enumerate(mskf.split(range(len(train_gps)), y)):
        if i == args.fold:
            break
    dev_gps = [train_gps[ind] for ind in dev_index]
    train_gps = [train_gps[ind] for ind in train_index]

    return train_gps, dev_gps


def kaggle2coco(kaggle_rle, h, w):
    if not len(kaggle_rle):
        return {'counts': [h * w], 'size': [h, w]}
    roll2 = np.roll(kaggle_rle, 2)
    roll2[:2] = 1

    roll1 = np.roll(kaggle_rle, 1)
    roll1[:1] = 0

    if h * w != kaggle_rle[-1] + kaggle_rle[-2] - 1:
        shift = 1
        end_value = h * w - kaggle_rle[-1] - kaggle_rle[-2] + 1
    else:
        shift = 0
        end_value = 0
    coco_rle = np.full(len(kaggle_rle) + shift, end_value)
    coco_rle[:len(coco_rle) - shift] = kaggle_rle.copy()
    coco_rle[:len(coco_rle) - shift:2] = (kaggle_rle - roll1 - roll2)[::2].copy()

    return {'counts': coco_rle.tolist(), 'size': [h, w]}


def read_img(path):
    return jpeg4py.JPEG(str(path)).decode()


def read_preds(path):
    return np.load(path)


def retrieve_img_mask(item, root, preds=False, binary=True):
    filename, group = item
    if preds:
        img = read_preds(root / (filename + '.npy')).transpose((1, 2, 0))
    else:
        img = read_img(root / filename)

    height, width = img.shape[:2]

    if binary:
        mask = np.zeros((height, width, 4), dtype='uint8')
    else:
        mask = np.zeros((height, width), dtype='uint8')
    for item in group.itertuples():
        if item.EncodedPixels != item.EncodedPixels:
            continue
        rle = kaggle2coco(list(map(int, item.EncodedPixels.split())), height, width)
        rle = mutils.frPyObjects(rle, height, width)
        if binary:
            mask[mutils.decode(rle) > 0, int(item.ClassId) - 1] = 255
        else:
            mask[mutils.decode(rle) > 0] = int(item.ClassId)

    return img, mask


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
