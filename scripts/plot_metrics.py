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

# ---- Identify all loss columns ----
loss_cols = [c for c in df.columns if c.startswith("loss")]

# ---- Create figure dynamically based on number of losses ----
n_losses = len(loss_cols)
n_cols = 2
n_rows = (n_losses + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()  # flatten in case of multiple rows

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

# If there are any unused axes, remove them
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved plot to {output_path}")
