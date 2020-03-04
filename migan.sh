export NVIM_LISTEN_ADDRESS=/tmp/nvimS9Bpnd/0
$1 train.py --name migan --netG miganlatentall\
    --dataset_mode danbooru2019 --dataroot /home/natsuki/danbooru2019 --use_hed \
    --no_html --tf_log --batchSize 16 \
    --niter 100 --label_nc 0 --save_epoch_freq 1 --continue_train

