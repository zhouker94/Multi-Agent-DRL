#!/bin/bash

cd agents/
for num in 6 8 10
do
    for w in 0.2 0.4 0.6 0.8 1.0
        do
            python ddpg_agent.py --n_agents=${num} --sustain_weight=${w}
        done
done
