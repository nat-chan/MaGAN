$1 train.py --name migan --netG miganlatentall\
    --dataset_mode danbooru2019 --dataroot /home/natsuki/danbooru2019 --use_hed \
    --no_html --tf_log --batchSize 16 \
    --niter 100 --label_nc 4 --no_instance --save_epoch_freq 1 --leak_low 3000 --leak_high 4000
#    --gpu_ids -1
# --continue_train

