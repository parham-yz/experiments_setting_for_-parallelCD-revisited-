#!/usr/bin/env bash
# Grid experiments with dynamic batching based on NUM_GPUS and TASKSPERCORE
# Batches are of size NUM_GPUS * TASKSPERCORE
# Each batch launches experiments in parallel and waits for their completion
# Includes trap to terminate background jobs on script exit


# --- Trap handler function ---
cleanup() {
  echo "" # Add a newline for cleaner output after the interrupt signal (^C)
  echo "Caught termination signal, stopping background jobs..."
  # Iterate through the PIDS of currently running jobs and terminate them
  for pid in "${PIDS[@]}"; do
    if ps -p "$pid" > /dev/null; then # Check if the process is still running
      echo "Terminating process $pid..."
      kill -TERM "$pid" 2>/dev/null # Send terminate signal, ignore errors if already exited
    fi
  done
  # Optional: wait for background processes to fully terminate
  # wait
  echo "Cleanup complete."
  exit 1 # Exit the script after cleanup
}

# --- Set trap for termination signals ---
# Trap INT (Interrupt, typically from Ctrl+C) and TERM (Terminate)
trap cleanup INT TERM

# Hyperparameters
REPORTS_DIR_NAME="ResNet34-bi-cifar100"  # You can change this value as needed
KS=(1 2 )
STEPS=(0.01 0.001 0.0001) # 5 step sizes
NUM_GPUS=8        # Total number of GPUs available
TASKSPERCORE=1         # Number of tasks to assign to each GPU per batch

BATCH_CHUNK_SIZE=$(( NUM_GPUS * TASKSPERCORE ))
ROUNDS=3500
BATCH_SIZE=256
REPORT_RATE=20

# --- Generate the list of all experiment configurations ---
EXPERIMENT_LIST=()
for STEP in "${STEPS[@]}"; do
  for K in "${KS[@]}"; do
    EXPERIMENT_LIST+=("${K}_${STEP}")
  done
done

TOTAL_EXPERIMENTS=${#EXPERIMENT_LIST[@]}
echo "Total experiments to run: $TOTAL_EXPERIMENTS"
echo "Batch chunk size: $BATCH_CHUNK_SIZE"
echo "Number of batches: $(( (TOTAL_EXPERIMENTS + BATCH_CHUNK_SIZE - 1) / BATCH_CHUNK_SIZE ))" # Ceiling division

echo "Cleaning up reports directory..."
mkdir -p "reports/$REPORTS_DIR_NAME"
rm -rf reports/"$REPORTS_DIR_NAME"/*

echo "Starting $REPORTS_DIR_NAME grid with dynamic batching..."

COUNTER=0 # Overall experiment counter
PIDS=()   # Declare PIDS array in a scope accessible to the trap handler

# Loop through the experiment list in chunks
for (( i=0; i<TOTAL_EXPERIMENTS; i+=BATCH_CHUNK_SIZE )); do
  # Clear PIDS for the new batch
  PIDS=()

  CURRENT_BATCH_EXPERIMENTS=("${EXPERIMENT_LIST[@]:i:BATCH_CHUNK_SIZE}")
  BATCH_SIZE_ACTUAL=${#CURRENT_BATCH_EXPERIMENTS[@]}
  BATCH_START_INDEX=$i



  echo "-- Starting batch $(( i / BATCH_CHUNK_SIZE + 1 )) (experiments $BATCH_START_INDEX to $(( BATCH_START_INDEX + BATCH_SIZE_ACTUAL - 1 )) out of $TOTAL_EXPERIMENTS) --"

  batch_task_counter=0 # Counter for tasks within the current batch

  # Loop through experiments in the current batch chunk
  for experiment_str in "${CURRENT_BATCH_EXPERIMENTS[@]}"; do
    # Split the string into K and STEP
    IFS=_ read -r current_k current_step <<< "$experiment_str"

    MODE="blockwise_sequential"
    if [ "$current_k" -eq 1 ]; then MODE="entire"; fi

    # Calculate GPU based on batch_task_counter and TASKSPERCORE
    GPU=$(( (batch_task_counter / TASKSPERCORE) % NUM_GPUS ))

    COUNTER=$(( COUNTER + 1 )) # Increment over all counter
    echo "[$COUNTER/$TOTAL_EXPERIMENTS] Launching K=$current_k, step_size=$current_step on GPU $GPU (Task index in batch: $batch_task_counter)"

    python3 -m src.dol1 \
      --model ResNet34-bi \
      --dataset_name cifar100 \
      --training_mode "$MODE" \
      --step_size "$current_step" \
      --batch_size "$BATCH_SIZE" \
      --rounds "$ROUNDS" \
      --K "$current_k" \
      --cuda_core "$GPU" \
      --communication_delay 0 \
      --report_sampling_rate "$REPORT_RATE" \
      --reports_dir "$REPORTS_DIR_NAME" &

    PIDS+=("$!")
    batch_task_counter=$(( batch_task_counter + 1 )) # Increment batch task counter

  done

  # Wait for all processes in the current batch chunk to finish
  echo "Waiting for batch to complete ($BATCH_SIZE_ACTUAL tasks)..."
  # The trap can interrupt this wait. The cleanup function will handle the background jobs.
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  echo "Batch complete."

done

echo "All $TOTAL_EXPERIMENTS CIFAR-100/ResNet18 experiments completed."
echo "Reports are in the $REPORTS_DIR_NAME/ directory."