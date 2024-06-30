#!/usr/bin/env bash

# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"  # specify your gpu id for training

stage=1  # start from -1 if you need to download data
stop_stage=1

config=configs/base.json  #
dir=exp/wangwen-mix  # training dir
test_audio=test_audio

# Please download data from https://www.data-baker.com/data/index/TNtts/, and
# set `raw_data_dir` to your data.
raw_data_dir=/data1/xiepengyuan/audio/tts/wangwen-mix
data=data/wangwen-mix
use_onnx=false

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  python local/prepare_data.py $raw_data_dir $data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  export MASTER_ADDR=localhost
  export MASTER_PORT=10085
  python vits/train.py -c $config -m $dir \
    --train_data $data/train.txt \
    --val_data $data/val.txt \
    --phone_table $data/phones.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python vits/export_onnx.py  \
    --checkpoint $dir/G_90000.pth \
    --cfg configs/base.json \
    --onnx_model $dir/G_90000.onnx \
    --phone_table data/phones.txt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  [ ! -d ${test_audio} ] && mkdir ${test_audio}
  if $use_onnx; then
    python vits/inference_onnx.py  \
      --onnx_model $dir/G_90000.onnx --cfg $config \
      --outdir $test_audio \
      --phone_table $data/phones.txt \
      --test_file $data/test.txt
  else
    python vits/inference.py  \
      --checkpoint $dir/G_90000.pth --cfg $config \
      --outdir $test_audio \
      --phone_table $data/phones.txt \
      --test_file $data/test.txt
  fi
fi
