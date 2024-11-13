#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
strategy="ddp"
upload_user="microsoft"
model_type="swin-large-patch4-window7-224-in22k"
is_crop=True
classification_type=4
precision=32
batch_size=32
epoch=10
model_detail="swin-large-patch4-window7-224-in22k"

python $path/prepare_upload.py --config-name=huggingface.yaml \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    is_crop=$is_crop \
    classification_type=$classification_type \
    precision=$precision \
    batch_size=$batch_size \
    epoch=$epoch \
    model_detail=$model_detail
