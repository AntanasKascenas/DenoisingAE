#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch


class Metric:
    """
    Registered metrics will print/tensorboard after a validation epoch is done. This means that BOTH training
    and validation batch results are accumulated until a validation epoch is performed by the trainer.
    """

    def __init__(self, name: str, val_prefix="val_", train_prefix="trn_"):
        self.name = name
        self.batch_key = "batch_" + name
        self.accumulation_key = "epoch_accumulation_" + name
        self.val_prefix = val_prefix
        self.train_prefix = train_prefix

    def calculate_batch(self, batch, batch_result):
        raise NotImplementedError("method not implemented.")

    def calculate_epoch(self, accumulation_list: list):
        raise NotImplementedError("method not implemented.")

    def to_dict(self, epoch_result):
        if isinstance(epoch_result, dict):
            return epoch_result
        else:
            return {self.name: epoch_result}

    def write_tensorboard(self, state, dict, prefix=""):
        for k, v in dict.items():
            state["writer"].add_scalar(f"{self.name}/{prefix}_{k}", v, state["epoch_no"])
        
    def print(self, state, dict, prefix=""):
        for k, v in dict.items():
            print(f"  {prefix}_{k}: {v:.6f}\t", end="")
        print("")

    def register(self, callback_dict, log=True, tensorboard=True, train=True, val=True):
        
        def track_batches(trigger, batch_key, accumulation_key, get_batch):
            def batch_fun(trainer):
                state = trainer.state
                with torch.no_grad():
                    if accumulation_key not in state:
                        state[accumulation_key] = []
                    state[accumulation_key].append(self.calculate_batch(*get_batch(state)))

            callback_dict[trigger].append(batch_fun)

        def calculate(trigger, accumulation_key, save_key):
            def epoch_fun(trainer):
                state = trainer.state
                with torch.no_grad():
                    if accumulation_key in state:
                        state[save_key] = self.calculate_epoch(state[accumulation_key])
                        state[accumulation_key] = []
                        if log:
                            self.print(state, self.to_dict(state[save_key]), prefix=save_key)
                        if tensorboard:
                            self.write_tensorboard(state, self.to_dict(state[save_key]), prefix=save_key)
            
            callback_dict[trigger].append(epoch_fun)

        train_batch_key = f"{self.train_prefix}{self.batch_key}"
        val_batch_key = f"{self.val_prefix}{self.batch_key}"
        train_accumulation_key = f"{self.train_prefix}{self.accumulation_key}"
        val_accumulation_key = f"{self.val_prefix}{self.accumulation_key}"
        train_key = f"{self.train_prefix}{self.name}"
        val_key = f"{self.val_prefix}{self.name}"

        if train:
            track_batches(trigger="after_train_step",
                        batch_key=train_batch_key,
                        accumulation_key=train_accumulation_key,
                        get_batch=lambda state: (state["train_batch"], state["train_batch_result"]))

            # Do train calculations just before val epoch with whatever has been accumulated
            calculate("before_val_epoch", train_accumulation_key, train_key)

        if val:
            track_batches(trigger="after_val_step",
                        batch_key=val_batch_key,
                        accumulation_key=val_accumulation_key,
                        get_batch=lambda state: (state["val_batch"], state["val_batch_result"]))

            calculate("after_val_epoch", val_accumulation_key, val_key)


class Loss(Metric):
    def __init__(self, loss, name="loss"):
        super().__init__(name)
        self.loss = loss

    def calculate_batch(self, batch, batch_results):
        with torch.no_grad():
            value = self.loss(batch_results, batch)
            shape = batch[0].shape
            # Take the number of pixels in the image as number of "elements"
            n = shape[0] * shape[-2] * shape[-1]
            return value, n

    def calculate_epoch(self, accumulation_list):
        with torch.no_grad():
            s = 0
            total = 0
            for v, n in accumulation_list:
                s += v * n
                total += n
            return (s / total).cpu().item()

