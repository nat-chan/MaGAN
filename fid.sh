#!/bin/bash
name=$1
name=$(basename ${name})
name=${name%.yml}
epoch=$2
gpu_ids=$3
[ -z "$gpu_ids" ]&&gpu_ids=0
results_dir=/data/natsuki/results

export CUDA_VISIBLE_DEVICES=${gpu_ids}
python evalute.py --conf=./parameters/${name}.yml --conf2=./parameters/test.yml --which_epoch=${epoch} --results_dir=${results_dir}
python pytorch-fid/fid_score.py --gpu ${gpu_ids} \
       ${results_dir}/${name}/test_${epoch}/images/real \
       ${results_dir}/${name}/test_${epoch}/images/synth \
       |sed 's/FID:  //'|tee ${results_dir}/${name}/test_${epoch}/fid
