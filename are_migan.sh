#!/bin/zsh

python3 evalute.py --name migan --netG miganlatentall\
    --dataset_mode danbooru2019 --dataroot /home/natsuki/danbooru2019 --use_hed --label_nc 0\
    --which_epoch 2
#    --gpu_ids 0,1,2,3,4,5,6,7  --batchSize 32\

