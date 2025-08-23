import time
import random
import logging
from tqdm import tqdm

# --- 1. Set up logging to a file ---
# This will save a clean record of epoch results without any special characters.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='training.log',
    filemode='w' # 'w' to overwrite the file each time, 'a' to append
)

# --- 2. Training Loop ---
num_epochs = 10
total_steps = 100

# Print the headers once at the beginning
HEADER = (
    f"{'Epoch':>10} {'GPU_mem':>10} {'box_loss':>10} {'cls_loss':>10} {'dfl_loss':>10}\n"
    f"{'':>10} {'Class':>10} {'Images':>10} {'Instances':>10} {'P':>10} {'R':>10} {'mAP50':>10}"
)
print(HEADER)
NUM_HEADER_LINES = 2

for epoch in range(num_epochs):
    # --- Inner loop with tqdm for steps ---
    progress_bar = tqdm(range(total_steps), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for step in progress_bar:
        # Simulate training step and update tqdm postfix
        loss = 3.5 - (epoch * 0.1) - (step / total_steps)
        progress_bar.set_postfix(loss=f"{loss:.4f}")
        time.sleep(0.01)

    # --- After inner loop, update the multi-line display ---
    # Move cursor up to overwrite the previous epoch's results
    # We move up one line for the previous data + one line for the empty line tqdm leaves
    # if epoch > 0:
    #     print("\x1b[2A", end="")

    # Clear the lines before writing new data
    # print("\x1b[K", end="\n") # Clear first line
    # print("\x1b[K", end="\n") # Clear second line
    # print("\x1b[2A", end="") # Move back up to the start

    # --- Prepare new data for display ---
    epoch_str = f"{epoch+1}/{num_epochs}"
    gpu_mem = f"{11.9 - random.random():.1f}G"
    box_loss = f"{3.5 - epoch * 0.1 - random.random() * 0.1:.4f}"
    cls_loss = f"{6.1 - epoch * 0.1 - random.random() * 0.1:.4f}"
    dfl_loss = f"{2.6 - epoch * 0.1 - random.random() * 0.1:.4f}"

    p_val = f"{min(0.7 + random.random() * 0.1, 0.95):.4f}"
    r_val = f"{min(0.05 + epoch * 0.02, 0.8):.4f}"
    map_val = f"{min(0.02 + epoch * 0.025, 0.75):.4f}"
    
    # --- Print the updated block ---
    epoch_line = f"{epoch_str:>10} {gpu_mem:>10} {box_loss:>10} {cls_loss:>10} {dfl_loss:>10}"
    val_line = f"{'':>10} {'all':>10} {'15':>10} {'70':>10} {p_val:>10} {r_val:>10} {map_val:>10}"
    print(epoch_line)
    print(val_line)

    # --- 3. Log the final, clean results to the file ---
    logging.info(f"Epoch {epoch+1} complete. Loss={box_loss}, mAP50={map_val}")

print("\nTraining finished! Check training.log for a summary. ✅")