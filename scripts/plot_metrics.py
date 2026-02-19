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
        if "split" in obj:  # skip config lines or malformed entries
            records.append(obj)

df = pd.DataFrame(records)

train_df = df[df["split"] == "train"].copy()
eval_df  = df[df["split"] == "eval"].copy()

# ---- Identify all loss columns with at least some non-null values ----
loss_cols = [c for c in df.columns if c.startswith("loss") and (
    train_df[c].notnull().any() or eval_df[c].notnull().any()
)]

# Add 1 for the Proj Norm + LR subplot
n_losses = len(loss_cols) + 1
n_cols = 2
n_rows = (n_losses + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

# ---- Plot all valid losses ----
for i, loss_name in enumerate(loss_cols):
    ax = axes[i]
    if loss_name in train_df.columns:
        ax.plot(train_df["step"], train_df[loss_name], label="Train")
    if loss_name in eval_df.columns:
        ax.plot(eval_df["step"], eval_df[loss_name], label="Eval")
    ax.set_title(loss_name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

# ---- Last subplot: Proj Norm + Learning Rate ----
ax = axes[len(loss_cols)]
if "proj_norm" in train_df.columns:
    ax.plot(train_df["step"], train_df["proj_norm"], label="Train Proj Norm")
if "proj_norm" in eval_df.columns:
    ax.plot(eval_df["step"], eval_df["proj_norm"], label="Eval Proj Norm")
ax.set_title("Proj Norm & Learning Rate")
ax.set_xlabel("Step")
ax.set_ylabel("Proj Norm")
ax.grid(True)

# Learning rate on secondary y-axis
ax2 = ax.twinx()
if "lr_proj" in train_df.columns:
    ax2.plot(train_df["step"], train_df["lr_proj"], linestyle="--", color="tab:orange", label="Train LR")
if "lr_proj" in eval_df.columns:
    ax2.plot(eval_df["step"], eval_df["lr_proj"], linestyle="--", color="tab:red", label="Eval LR")
ax2.set_ylabel("Learning Rate")

# Combine legends from both axes
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="best")

# Remove unused axes
for j in range(len(loss_cols) + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved plot to {output_path}")
