#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from math import ceil
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from denoising import denoising
from data import BrainDataset
from cc_filter import connected_components_3d


def eval_anomalies_batched(trainer, dataset, get_scores, batch_size=32, threshold=None, get_y=lambda batch: batch[1],
                           return_dice=False, filter_cc=False):
    def dice(a, b):
        num = 2 * (a & b).sum()
        den = a.sum() + b.sum()

        den_float = den.float()
        den_float[den == 0] = float("nan")

        return num.float() / den_float

    y_true_ = torch.zeros(128 * 128 * len(dataset), dtype=torch.half)
    y_pred_ = torch.zeros(128 * 128 * len(dataset), dtype=torch.half)

    n_batches = int(ceil(len(dataset) / batch_size))
    i = 0
    for batch_idx in range(n_batches):
        batch_items = [dataset[x] for x in
                       range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))]
        collate = torch.utils.data._utils.collate.default_collate
        batch = collate(batch_items)

        batch_y = get_y(batch)

        with torch.no_grad():
            anomaly_scores = get_scores(trainer, batch)

        y_ = (batch_y.view(-1) > 0.5)
        y_hat = anomaly_scores.reshape(-1)
        # Use half precision to save space in RAM. Want to evaluate the whole dataset at once.
        y_true_[i:i + y_.numel()] = y_.half()
        y_pred_[i:i + y_hat.numel()] = y_hat.half()
        i += y_.numel()

    ap = average_precision_score(y_true_, y_pred_)

    if return_dice:
        dice_thresholds = [x / 1000 for x in range(1000)] if threshold is None else [threshold]
        with torch.no_grad():
            y_true_ = y_true_.to(trainer.device)
            y_pred_ = y_pred_.to(trainer.device)
            dices = [dice(y_true_ > 0.5, y_pred_ > x).cpu().item() for x in tqdm(dice_thresholds)]
        max_dice, threshold = max(zip(dices, dice_thresholds), key=lambda x: x[0])

        if filter_cc:
            # Now that we have the threshold we can do some filtering and recalculate the Dice
            i = 0
            y_true_ = torch.zeros(128 * 128 * len(dataset), dtype=torch.bool)
            y_pred_ = torch.zeros(128 * 128 * len(dataset), dtype=torch.bool)

            for pd in dataset.patient_datasets:
                batch_items = [pd[x] for x in range(len(pd))]
                collate = torch.utils.data._utils.collate.default_collate
                batch = collate(batch_items)

                batch_y = get_y(batch)

                with torch.no_grad():
                    anomaly_scores = get_scores(trainer, batch)

                # Do CC filtering:
                anomaly_scores_bin = anomaly_scores > threshold
                anomaly_scores_bin = connected_components_3d(anomaly_scores_bin.squeeze(dim=1)).unsqueeze(dim=1)

                y_ = (batch_y.view(-1) > 0.5)
                y_hat = anomaly_scores_bin.reshape(-1)
                y_true_[i:i + y_.numel()] = y_
                y_pred_[i:i + y_hat.numel()] = y_hat
                i += y_.numel()

            with torch.no_grad():
                y_true_ = y_true_.to(trainer.device)
                y_pred_ = y_pred_.to(trainer.device)
                post_cc_max_dice = dice(y_true_, y_pred_).cpu().item()

            return ap, max_dice, threshold, post_cc_max_dice
        return ap, max_dice, threshold
    return ap


def evaluate(id: str = "model", split: str = "test", use_cc: bool = True):
    trainer = denoising(id, data=None, lr=0.0001, depth=4,
                        wf=6, noise_std=0.2, noise_res=16)  # Noise parameters don't matter during evaluation.

    trainer.load(id)

    dataset = BrainDataset(dataset="brats2021", split=split, n_tumour_patients=None, n_healthy_patients=0)

    results = eval_anomalies_batched(trainer, dataset=dataset, get_scores=trainer.get_scores, return_dice=True,
                                     filter_cc=use_cc)

    print(f"AP: {results[0]}")
    print(f"max Dice: {results[1]}")
    if use_cc:
        print(f"max Dice post CC: {results[3]}")
    print(f"Optimal threshold: {results[2]}")

    return results


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--identifier", required=True, type=str, help="identifier for model to evaluate")
    parser.add_argument("-s", "--split", required=True, type=str, help="'train', 'val' or 'test'")
    parser.add_argument("-cc", "--use_cc", required=False, type=bool, default=True, help="Whether to use connected component filtering.")

    args = parser.parse_args()

    evaluate(id=args.identifier,
             split=args.split,
             use_cc=args.use_cc)







