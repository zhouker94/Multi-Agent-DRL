#!/bin/bash

cd agents/
for num in 1 2 4 6 8 10
do
    for w in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            python main_loop.py --n_agent=${num} --sustainable_weight=${w}
        done
done
