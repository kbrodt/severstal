import argparse
import copy as c
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help='Path to models to average')

    return parser.parse_args()


def main():
    args = parse_args()

    path_to_models = Path(args.path)
    sds = []
    for p in path_to_models.glob('*.pth'):
        sd = torch.load(p)
        print(p, sd['history']['dice']['dev'][-1])
        sds.append(sd['state_dict'])

    sd_a = c.deepcopy(sds[0])
    for k in sd_a:
        for m in sds[1:]:
            sd_a[k] += m[k]
    for k in sd_a:
        sd_a[k] /= len(sds)

    sd['state_dict'] = sd_a

    path_to_save = Path(str(path_to_models) + '_ave')
    print(path_to_save)
    if not path_to_save.exists():
        path_to_save.mkdir(parents=True)
    torch.save(sd, path_to_save / 'averaged.pth')


if __name__ == '__main__':
    main()
