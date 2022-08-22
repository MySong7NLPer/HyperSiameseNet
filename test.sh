#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train_matching.py --mode=test --encoder=bert --save_path=/data/nick/cikm/hyperbolic/xxx --gpus=0


