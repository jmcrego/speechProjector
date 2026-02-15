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

# 4) Audio Norm
ax = axes[1, 1]
ax.plot(train_df["step"], train_df["audio_norm"], label="Train")
ax.plot(eval_df["step"], eval_df["audio_norm"], label="Eval")
ax.set_title("Audio Embedding Norm")
ax.set_xlabel("Step")
ax.set_ylabel("Norm")
ax.legend()

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved plot to {output_path}")
