#!/bin/bash
# Monitor extraction and restart if stopped

LOG=/home/ps/Documents/code/EventGPT/feasible/feature_alignment/extraction_resume.log
SCRIPT=/home/ps/Documents/code/EventGPT/feasible/feature_alignment/extract_hidden_states.py
WORKDIR=/home/ps/Documents/code/EventGPT

check_and_restart() {
    # Check if extraction is running
    if ! pgrep -f "extract_hidden_states.py" > /dev/null; then
        echo "[$(date)] Extraction stopped! Checking last status..."
        tail -5 $LOG
        
        # Check if completed
        if grep -q "Extraction complete" $LOG; then
            echo "[$(date)] Extraction completed successfully!"
            return 1
        fi
        
        echo "[$(date)] Restarting extraction..."
        cd $WORKDIR
        source /home/ps/anaconda3/etc/profile.d/conda.sh
        conda activate egpt
        nohup python $SCRIPT \
            --split train \
            --max_samples 5208 \
            --max_questions 10 \
            --max_new_tokens 50 \
            --output_dir feasible/feature_alignment/hidden_states \
            --duration 1s \
            --quant 4bit \
            --save_interval 100 \
            --resume \
            > $LOG 2>&1 &
        echo "[$(date)] Restarted with PID: $!"
    else
        # Show current progress
        tail -1 $LOG | grep -o "extracted=[0-9]*" | head -1
    fi
    return 0
}

echo "[$(date)] Starting extraction monitor..."
while check_and_restart; do
    sleep 600  # 10 minutes
done
