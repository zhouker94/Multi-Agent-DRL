#!/bin/bash
for num in 2 4 6 8 10
do
    for w in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            python ddpg_main_loop.py --n_agents=${num} --sustainable_weight=${w}
        done
done
