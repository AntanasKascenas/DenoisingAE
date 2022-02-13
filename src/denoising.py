#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from typing import Union, Optional
from collections import defaultdict
from functools import partial
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from training import simple_train_step, simple_val_step
from metrics import Loss
from trainer import Trainer
from data_descriptor import DecathlonBrainAEDescriptor, DataDescriptor
from utilities import median_pool, ModelSaver
from unet import UNet


def denoising(identifier: str, data: Optional[Union[str, DataDescriptor]] = None, lr=0.001, depth=4, wf=7, n_input=4, noise_std=0.2, noise_res=16):
    device = torch.device("cuda")

    def noise(x):

        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[128, 128])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(128))
        roll_y = random.choice(range(128))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask # Only apply the noise in the foreground.
        res = x + ns

        return res

    def get_scores(trainer, batch, median_f=True):
        x = batch[0]
        trainer.model = trainer.model.eval()
        with torch.no_grad():
            # Assume it's in batch shape
            clean = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]).clone().to(trainer.device)
            mask = clean.sum(dim=1, keepdim=True) > 0.01

            # Erode the mask a bit to remove some of the reconstruction errors at the edges.
            mask = (F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.95)

            res = trainer.model(clean)

            err = ((clean - res) * mask).abs().mean(dim=1, keepdim=True)
            if median_f:
                err = median_pool(err, kernel_size=5, stride=1, padding=2)

        return err.cpu()

    def loss_f(trainer, batch, batch_results):

        y = batch[1]
        mask = batch[1].sum(dim=1, keepdim=True) > 0.01

        return (torch.pow(batch_results - y, 2) * mask.float()).mean()

    def forward(trainer, batch):
        batch[1] = batch[0] # Clean image is the "target"
        batch[0] = noise(batch[0].clone())

        return trainer.model(batch[0])

    model = UNet(in_channels=n_input, n_classes=n_input, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)

    train_step = partial(simple_train_step, forward=forward, loss_f=loss_f)
    val_step = partial(simple_val_step, forward=forward, loss_f=loss_f)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.00001)
    callback_dict = defaultdict(list)

    model_saver = ModelSaver(path=Path(__file__).resolve().parent.parent / "saved_models" / f"{identifier}.pt")
    model_saver.register(callback_dict)

    Loss(lambda batch_res, batch: loss_f(trainer, batch, batch_res)).register(callback_dict, log=True, tensorboard=True, train=True, val=True)

    trainer = Trainer(model=model,
                      train_dataloader=None,
                      val_dataloader=None,
                      optimiser=optimiser,
                      train_step=train_step,
                      val_step=val_step,
                      callback_dict=callback_dict,
                      device=device,
                      identifier=identifier)

    trainer.noise = noise
    trainer.get_scores = get_scores
    trainer.set_data(data)
    trainer.reset_state()


    trainer.lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=100)

    def update_learning_rate(trainer):
        trainer.lr_scheduler.step()

    trainer.callback_dict["after_train_epoch"].append(update_learning_rate)

    return trainer


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--identifier", type=str, default="model", help="model name.")
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed.")

    args = parser.parse_args()

    dd = DecathlonBrainAEDescriptor(dataset="brats2021", n_train_patients=None, n_val_patients=None,
                                    seed=args.seed, batch_size=16, )

    trainer = denoising(args.identifier, data=dd, lr=0.0001, depth=4,
                        wf=6, noise_std=args.noise_std, noise_res=args.noise_res)

    trainer.train(epoch_len=32, max_epochs=2100, val_epoch_len=32)
