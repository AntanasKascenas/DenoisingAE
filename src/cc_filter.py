"""
From: https://github.com/FeliMe/brain_sas_baseline

MIT License

Copyright (c) 2021 Felix Meissen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import numpy as np
from skimage.measure import label, regionprops


def connected_components_3d(volume):


    is_batch = True
    is_torch = torch.is_tensor(volume)
    if is_torch:
        volume = volume.numpy()
    if volume.ndim == 3:
        volume = np.expand_dims(volume, axis=0)
        is_batch = False

    # shape [b, d, h, w], treat every sample in batch independently
    pbar = range(len(volume))
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop['filled_area'] <= 20:
                volume[i, cc_volume == prop['label']] = 0

    if not is_batch:
        volume = volume.squeeze(0)
    if is_torch:
        volume = torch.from_numpy(volume)
    return volume