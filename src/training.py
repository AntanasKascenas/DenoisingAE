#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch


def detach(x):
    if isinstance(x, (tuple, list)):
        return [(z.detach() if isinstance(z, torch.Tensor) else z) for z in x]
    else:
        return x.detach()


def simple_train_step(trainer, forward, loss_f):
    state = trainer.state
    accumulation_steps = state["accumulation_steps"] # Accumulate gradients and only update model parameters every `accumulation_steps`th step.

    if state["train_it"] % accumulation_steps == 0:
        trainer.optimiser.zero_grad()

    batch = state["train_batch"]
    batch_result = forward(trainer, batch)
    # Want to detach so that we don't accumulate GPU memory as we go through batches.
    state["train_batch_result"] = detach(batch_result)
    loss = loss_f(trainer, batch, batch_result)

    # Check whether loss is NaN, if so, stop and raise an Exception:
    with torch.no_grad():
        if torch.isnan(loss).sum() >= 1:
            raise ValueError("NaN loss, stopping...")

    if loss.requires_grad: # Check whether gradients are disabled (e.g. during evaluation).
        loss.backward()
        if state["train_it"] % accumulation_steps == accumulation_steps - 1:
            torch.nn.utils.clip_grad.clip_grad_norm_(trainer.model.parameters(), max_norm=0.5)
            trainer.optimiser.step()


def simple_val_step(trainer, forward, loss_f):

    state = trainer.state
    with torch.no_grad():
        y_pred = forward(trainer, state["val_batch"])
    state["val_batch_result"] = y_pred



