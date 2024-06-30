# -*- coding: utf-8 -*-

import torch


def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def general_padding(item, item_len, max_len, pad_value=0):
    if item_len < max_len:
        item = torch.nn.functional.pad(item, (0, max_len - item_len), value=pad_value)
    return item
