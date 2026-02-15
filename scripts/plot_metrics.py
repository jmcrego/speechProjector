import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <log_file.jsonl>")
    sys.exit(1)

# ---- Path to your log file ----
log_file = sys.argv[1]
output_file = log_file + ".png"

steps = []
loss = []
loss_cos = []
loss_mse_txt = []
audio_norm = []
text_norm = []
grad_norm = []
lr_proj = []

with open(log_file, "r") as f:
    for line in f:
        data = json.loads(line)
        if data["split"] != "train":
            continue

        steps.append(data["step"])
        loss.append(data["loss"])
        loss_cos.append(data["loss_cos"])
        loss_mse_txt.append(data["loss_mse_txt"])
        audio_norm.append(data["audio_norm"])
        text_norm.append(data["text_norm"])
        grad_norm.append(data["proj_grad_norm"])
        lr_proj.append(data["lr_proj"])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# -----------------------
# 1) Total Loss
# -----------------------
axs[0, 0].plot(steps, loss)
axs[0, 0].set_title("Total Loss")
axs[0, 0].set_xlabel("Step")

# -----------------------
# 2) Cosine + MSE
# -----------------------
axs[0, 1].plot(steps, loss_cos)
axs[0, 1].plot(steps, loss_mse_txt)
axs[0, 1].set_title("Cosine & MSE (text)")
axs[0, 1].set_xlabel("Step")

# -----------------------
# 3) Embedding Norms
# -----------------------
axs[1, 0].plot(steps, audio_norm)
axs[1, 0].plot(steps, text_norm)
axs[1, 0].set_title("Embedding Norms")
axs[1, 0].set_xlabel("Step")

# -----------------------
# 4) Grad Norm + Learning Rate
# -----------------------
ax1 = axs[1, 1]
ax1.plot(steps, grad_norm)
ax1.set_title("Projector Grad Norm & LR")
ax1.set_xlabel("Step")
ax1.set_ylabel("Grad Norm")

ax2 = ax1.twinx()
ax2.plot(steps, lr_proj)
ax2.set_ylabel("Learning Rate")

plt.tight_layout()
plt.savefig(output_file, dpi=300)