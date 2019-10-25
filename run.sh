#!/bin/bash

# train models
python -m src.train --encoder efficientnet-b3 --opt adam --lr 0.0005 --batch-size 8 --cls --n-folds 5 --fold 0
python -m src.train --encoder efficientnet-b4 --opt adam --lr 0.0005 --batch-size 8 --cls --n-folds 5 --fold 1
python -m src.train --encoder efficientnet-b5 --opt adam --lr 0.0005 --batch-size 8 --cls --n-folds 5 --fold 2
python -m src.train --encoder efficientnet-b6 --opt adam --lr 0.0005 --batch-size 8 --cls --n-folds 5 --fold 3
python -m src.train --encoder efficientnet-b5 --opt adam --lr 0.0005 --batch-size 8 --cls --n-folds 5 --fold 4

# average weights of `last` and `best` models
python -m src.average --path efficientnet-b3_b8_adam_lr0.0005_c1_fold0
python -m src.average --path efficientnet-b4_b8_adam_lr0.0005_c1_fold1
python -m src.average --path efficientnet-b5_b8_adam_lr0.0005_c1_fold2
python -m src.average --path efficientnet-b6_b8_adam_lr0.0005_c1_fold3
python -m src.average --path efficientnet-b5_b8_adam_lr0.0005_c1_fold4

# predict oof masks to tune thresholds
python -m src.predict --load efficientnet-b3_b8_adam_lr0.0005_c1_fold0_ave/averaged.pth --save brave5
python -m src.predict --load efficientnet-b4_b8_adam_lr0.0005_c1_fold1_ave/averaged.pth --save brave5
python -m src.predict --load efficientnet-b5_b8_adam_lr0.0005_c1_fold2_ave/averaged.pth --save brave5
python -m src.predict --load efficientnet-b6_b8_adam_lr0.0005_c1_fold3_ave/averaged.pth --save brave5
python -m src.predict --load efficientnet-b5_b8_adam_lr0.0005_c1_fold4_ave/averaged.pth --save brave5

# find best threshes on oof masks
python -m src.thresh_search