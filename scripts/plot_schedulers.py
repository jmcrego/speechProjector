import torch
import matplotlib.pyplot as plt
from transformers import get_scheduler

# Dummy model parameter
model = torch.nn.Linear(10, 1)

# Training configuration
num_training_steps = 1000
num_warmup_steps = 100
base_lr = 5e-4

scheduler_names = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
]

num_schedulers = len(scheduler_names)

fig, axes = plt.subplots(
    nrows=num_schedulers,
    ncols=1,
    figsize=(10, 2.5 * num_schedulers),
    sharex=True
)

# If only one scheduler, axes is not a list
if num_schedulers == 1:
    axes = [axes]

for ax, name in zip(axes, scheduler_names):
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    scheduler = get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    lrs = []

    for step in range(num_training_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    ax.plot(lrs)
    ax.set_title(name)
    ax.set_ylabel("LR")
    ax.grid(True)

axes[-1].set_xlabel("Training Steps")

plt.tight_layout()

output_path = "lr_schedulers_vertical.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to {output_path}")
