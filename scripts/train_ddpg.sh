#!/bin/bash

for num in 1 2 4 6 8 10
do
    for w in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            mad-experiment --model "DDPG" --num_agent ${num} --sustainable_weight ${w} --run_mode "train"
        done
done
