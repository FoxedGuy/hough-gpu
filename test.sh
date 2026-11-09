#!/bin/bash

mkdir -p results

for repeat in {1..10}; do
    for block in 128 256 512 1024; do

        echo "Test: repeat=$repeat block=$block"

        ./hough_gpu_streams \
            horizontal_1024_160.png \
            500 \
            20000 \
            custom \
            "$block" \
            > "results/block_${block}_run_${repeat}.txt"

        mv power_custom.csv \
           "results/power_${block}_run_${repeat}.csv"

    done
done