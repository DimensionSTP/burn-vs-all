#!/bin/bash

path="src/preprocessing"
split_ratio=1e-1

python $path/split_metadata.py \
    split_ratio=$split_ratio