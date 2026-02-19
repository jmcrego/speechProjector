import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <log_file.jsonl>")
    sys.exit(1)

# ---- Path to your log file ----
log_path = sys.argv[1]
output_path = log_path + ".png"

# ---- Load JSONL ----
records = []
with open(log_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        if "split" in obj:  # skip config line
            records.append(obj)

df = pd.DataFrame(records)

train_df = df[df["split"] == "train"].copy()
eval_df  = df[df["split"] == "eval"].copy()

# ---- Create figure with 4 subplots ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Total Loss
ax = axes[0, 0]
ax.plot(train_df["step"], train_df["loss"], label="Train")
ax.plot(eval_df["step"], eval_df["loss"], label="Eval")
ax.set_title("Total Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.legend()

# 2) Cosine Loss
ax = axes[0, 1]
ax.plot(train_df["step"], train_df["loss_cos"], label="Train")
ax.plot(eval_df["step"], eval_df["loss_cos"], label="Eval")
ax.set_title("Cosine Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.legend()

# 3) MSE Losses
ax = axes[1, 0]
ax.plot(train_df["step"], train_df["loss_mse_txt"], label="Train MSE txt")
ax.plot(eval_df["step"], eval_df["loss_mse_txt"], label="Eval MSE txt")
ax.plot(train_df["step"], train_df["loss_mse_pad"], linestyle="--", label="Train MSE pad")
ax.plot(eval_df["step"], eval_df["loss_mse_pad"], linestyle="--", label="Eval MSE pad")
ax.set_title("MSE Loss")
ax.set_xlabel("Step")
ax.set_ylabel("MSE")
ax.legend()

# 4) Audio Norm + Learning Rate
ax = axes[1, 1]

# Audio norm (left axis)
ax.plot(train_df["step"], train_df["proj_norm"], label="Train Proj Norm")
ax.plot(eval_df["step"], eval_df["proj_norm"], label="Eval Proj Norm")
ax.set_title("Proj Norm & Learning Rate")
ax.set_xlabel("Step")
ax.set_ylabel("Proj Norm")
# Learning rate (right axis)
ax2 = ax.twinx()
if "lr_proj" in train_df.columns:
    ax2.plot(train_df["step"], train_df["lr_proj"], linestyle="--", label="Train LR")
# if "lr_proj" in eval_df.columns:
#     ax2.plot(eval_df["step"], eval_df["lr_proj"], linestyle="--", label="Eval LR")
ax2.set_ylabel("Learning Rate")

# Combine legends from both axes
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved plot to {output_path}")
