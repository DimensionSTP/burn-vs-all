#!/bin/bash

is_tuned="untuned"
strategy="ddp"
model_type="swin_large_patch4_window7_224"
pretrained="pretrained"
is_crop=True
classification_types="0 1 2 3 4"
precision=32
batch_size=32
workers_ratio=8
use_all_workers=False
epochs="9 10"

for epoch in $epochs
do
    for classification_type in $classification_types
    do
        python main.py --config-name=timm.yaml \
            mode=test \
            is_tuned=$is_tuned \
            strategy=$strategy \
            model_type=$model_type \
            pretrained=$pretrained \
            is_crop=$is_crop \
            classification_type=$classification_type \
            precision=$precision \
            batch_size=$batch_size \
            workers_ratio=$workers_ratio \
            use_all_workers=$use_all_workers \
            epoch=$epoch
    done
done

is_tuned="untuned"
strategy="ddp"
upload_user="microsoft"
model_type="swin-large-patch4-window7-224-in22k"
is_crop=True
classification_types="0 1 2 3 4"
precision=32
batch_size=32
workers_ratio=8
use_all_workers=False
epochs="9 10"

for epoch in $epochs
do
    for classification_type in $classification_types
    do
        python main.py --config-name=huggingface.yaml \
            mode=test \
            is_tuned=$is_tuned \
            strategy=$strategy \
            upload_user=$upload_user \
            model_type=$model_type \
            is_crop=$is_crop \
            classification_type=$classification_type \
            precision=$precision \
            batch_size=$batch_size \
            workers_ratio=$workers_ratio \
            use_all_workers=$use_all_workers \
            epoch=$epoch
    done
done