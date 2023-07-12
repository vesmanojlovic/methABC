#!/bin/bash

mkdir $1/methylation

for file in $1/simulations/*

do
    echo "Processing run $(basename ${file})."
    ./generate_inter-deme.R $file $1/methylation/$(basename ${file}).csv
    echo "Finished run $(basename ${file})."
done
echo "Finished all runs."