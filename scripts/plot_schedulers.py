import torch
import matplotlib.pyplot as plt
from transformers import get_scheduler

# Dummy model parameter (required for optimizer)
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

plt.figure(figsize=(10, 6))

for name in scheduler_names:
    # Recreate optimizer for each scheduler
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

    plt.plot(lrs, label=name)

plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("Comparison of HuggingFace LR Schedulers")
plt.legend()
plt.grid(True)

# Save instead of show
output_path = "lr_schedulers.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to {output_path}")
