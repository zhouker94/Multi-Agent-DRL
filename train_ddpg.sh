#!/bin/bash

cd agents/
for num in 10
do
    for w in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
        do
            python ddpg_agent.py --n_agents=${num} --sustainable_weight=${w}
        done
done
