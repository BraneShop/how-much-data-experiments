#!/usr/bin/env bash

for k in "" "-augx10"; do
  for i in 1 3 5 10 20 30 40 50 60 70 80 90 100; do
    python label_folder.py                   \
        --folder holdout                     \
        --prefix experiments_$i$k            \
        --graph ckpts/$i$k/graph.pb
  done
done
