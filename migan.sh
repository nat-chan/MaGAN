$1 train.py --name migan --netG miganlatentall\
    --dataset_mode danbooru2019 --dataroot /home/natsuki/danbooru2019 --use_hed \
    --no_html --tf_log --batchSize 32 \
    --niter 100 --label_nc 4 --no_instance --save_epoch_freq 1 --leak_low 16 --leak_high 512 \
    --gpu_ids 0,1,2,3,4,5,6,7
# https://github.com/pfnet/PaintsChainer/issues/99
# --continue_train

