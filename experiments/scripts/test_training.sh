#!/bin/bash
cd /home2/tathagato/summarization/MACSum/experiments

torchrun --standalone --nproc_per_node=4 trainer.py > debug.txt