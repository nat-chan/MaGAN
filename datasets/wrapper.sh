#!/bin/bash

for i in {1..9}0; do
    echo $i
    export ratio=$i
    export basename="./coco_stuff_hed/val_hed"
    suffix="x1"
    dstdir="${basename}${ratio}${suffix}"
    mkdir -p "${dstdir}"
    listnum=`ls ${basename}|sed 's/\.png//g'|wc -l`
    data="${dstdir}/data"
    P=20
    L=$((listnum/P))
    ls ${basename}|sed 's/\.png//g'|xargs -P $P -L $L ./a.out|tee "$data"| ./tq.py $listnum
done
