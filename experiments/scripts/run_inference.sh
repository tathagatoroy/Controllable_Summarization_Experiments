#!/bin/bash

source /home2/tathagato/miniconda3/bin/activate roy

echo "starting  gpu 1 in the background"
./gpu_1.sh > /home2/tathagato/summarization/MACSum/experiments/logs/gpu_1.log 2>&1 &
#get pid 
pid_1=$!

echo "starting  gpu 2 in the background"
./gpu_2.sh > /home2/tathagato/summarization/MACSum/experiments/logs/gpu_2.log 2>&1 &
#get pid
pid_2=$!

echo "starting gpu in the background"
./gpu_3.sh > /home2/tathagato/summarization/MACSum/experiments/logs/gpu_3.log  2>&1 &
#get pid
pid_3=$!

echo "starting  gpu 4 in the background"
./gpu_4.sh > /home2/tathagato/summarization/MACSum/experiments/logs/gpu_4.log 2>&1 &
#get pid
pid_4=$!
#wait for all pid
wait $pid_1 $pid_2 $pid_3 $pid_4
# wait $pid_1 $pid_2 
echo "all done"
