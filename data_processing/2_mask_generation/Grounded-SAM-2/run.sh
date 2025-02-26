#!/bin/bash

# Base command
BASE_CMD="python -u 2_mask_generation.py --output_dir /data_processing/assets/2_mask_generation --images_folder /data_processing/assets/0_source_img --json_folder /data_processing/assets/2_mask_generation"


for i in {0..9}
do
    LOG_FILE="log/mask_range_$i.log"
    nohup $BASE_CMD --range $i > $LOG_FILE 2>&1 &
    echo "Started range $i with PID $! and logging to $LOG_FILE"
done
