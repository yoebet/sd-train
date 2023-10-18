from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F


class UpperCrop(RandomCrop):

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:

        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        wd = w - tw
        hd = h - th
        wp = int(wd / 4)
        hp = int(hd / 4)
        top = torch.randint(0, (hp + hp) + 1, size=(1,)).item()
        left = torch.randint(wp, (wd - wp) + 1, size=(1,)).item()
        # top = 0
        # left = (w - tw) / 2
        return top, left, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
