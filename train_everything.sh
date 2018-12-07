#!/usr/bin/env zsh

for k in "" "-augx10"; do;
  # Normal kind
  for i in 5 10 20 30 40 50 60 70 80 90 100; do; 
    python retrain.py                        \
        --summaries_dir /tmp/retrain_logs    \
        --image_dir experiments/$i$k         \
        --log_level 1                        \
        --summaries_dir ckpts/$i$k           \
        --output_graph ckpts/$i$k/graph.pb;
  done;

  # Not enough data to have validation sets.
  # Note: In my experimenting, it didn't make much difference
  #       if I left the number of steps at 4k (default) or made
  #       it 300.
  for i in 1 3; do; 
    python retrain.py                        \
        --summaries_dir /tmp/retrain_logs    \
        --image_dir experiments/$i$k         \
        --log_level 1                        \
        --summaries_dir ckpts/$i$k           \
        --output_graph ckpts/$i$k/graph.pb   \
        --how_many_traning_steps 300         \
        --validation_percentage 0            \
        --testing_percentage 0;
  done;
done;
