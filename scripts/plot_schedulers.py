import torch
import matplotlib.pyplot as plt
from transformers import get_scheduler


def get_lrs(scheduler_name, model, base_lr, warmup_steps, total_steps):
    """
    Create optimizer + scheduler and return LR values over all steps.
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    scheduler_kwargs = {}

    # Make cosine_with_restarts visibly different
    if scheduler_name == "cosine_with_restarts":
        scheduler_kwargs["scheduler_specific_kwargs"] = {"num_cycles": 5}

    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        **scheduler_kwargs
    )

    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    return lrs


def main():

    # Dummy model (just to create optimizer)
    model = torch.nn.Linear(10, 1)

    # Training config
    total_steps = 1000
    warmup_steps = 100
    base_lr = 5e-4

    scheduler_names = [
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]

    num_plots = len(scheduler_names)

    fig, axes = plt.subplots(
        nrows=num_plots,
        ncols=1,
        figsize=(10, 2.5 * num_plots),
        sharex=True
    )

    if num_plots == 1:
        axes = [axes]

    for ax, name in zip(axes, scheduler_names):

        lrs = get_lrs(
            scheduler_name=name,
            model=model,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        ax.plot(lrs)
        ax.set_title(name)
        ax.set_ylabel("LR")
        ax.grid(True)

    axes[-1].set_xlabel("Training Steps")

    plt.tight_layout()

    output_file = "lr_schedulers_vertical.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
