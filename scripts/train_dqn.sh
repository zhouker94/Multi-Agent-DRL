#!/bin/bash

for num in 10
do
    for w in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
        do
            mad-experiment --model "DQN" --num_agent ${num} --sustainable_weight ${w} --run_mode "train"
        done
done
