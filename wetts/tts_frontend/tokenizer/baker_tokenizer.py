# -*- coding: utf-8 -*-

import re
import os
import numpy as np
from typing import List, Optional


class BakerTokenizer:
    def __init__(
        self,
        phonemes: Optional[List[str]] = None,
        tones: Optional[List[str]] = None,
        poses: Optional[List[str]] = None,
        prosodys: Optional[List[str]] = None,
    ):
        self.phoneme2idx = {s: i for i, s in enumerate(phonemes)}
        self.tone2idx = {s: i for i, s in enumerate(tones)}
        self.pos2idx = {s: i for i, s in enumerate(poses)}
        self.prosody2idx = {s: i for i, s in enumerate(prosodys)}

        self.idx2phoneme = {i: s for i, s in enumerate(phonemes)}
        self.idx2tone = {i: s for i, s in enumerate(tones)}
        self.idx2pos = {i: s for i, s in enumerate(poses)}
        self.idx2prosody = {i: s for i, s in enumerate(prosodys)}

    def __call__(self, text):
        phonemes = []
        tones = []
        prosodys = []
        poses = []
        tokens = re.split(r"\s+", text)
        tokens = [token for token in tokens if token]
        tokens = ["#4"] + tokens
        for token in tokens:
            if re.fullmatch(r"#\d|\W", token):
                # if re.fullmatch(r"#\d|\W", token) and len(prosodys) > 1:
                if re.fullmatch(r"#\d", token) and len(prosodys) > 1:
                    prosodys[-1] = self.prosody2idx[token]
                if re.fullmatch(r"#[34]", token):
                    phonemes.append(self.phoneme2idx[token])
                    tones.append(self.tone2idx["O"])
                    poses.append(self.pos2idx["O"])
                    prosodys.append(self.prosody2idx[token])
            else:  # pinyin
                tone = token[-1]
                phoneme_str = self.pinyin2phoneme(token)
                ps = re.split(r"\s+", phoneme_str)
                assert 1 <= len(ps) <= 2
                if len(ps) == 1:
                    phoneme = ps[0][:-1]
                    phonemes.append(self.phoneme2idx[phoneme])
                    tones.append(self.tone2idx[tone])
                    if token == "er5":
                        if poses[-1] == self.pos2idx["S"]:
                            poses[-1] = self.pos2idx["B"]
                        else:
                            poses[-1] = self.pos2idx["I"]
                        poses.append(self.pos2idx["E"])
                    else:
                        poses.append(self.pos2idx["S"])
                    prosodys.append(self.prosody2idx["#0"])
                else:
                    phoneme = ps[0]
                    phonemes.append(self.phoneme2idx[phoneme])
                    if re.fullmatch(r"\^", phoneme):
                        tones.append(self.tone2idx["sm0"])
                    else:
                        tones.append(self.tone2idx["sm"])
                        # tones.append(tone2idx[f"sm{tone}"])
                    poses.append(self.pos2idx["B"])
                    prosodys.append(self.prosody2idx["O"])
                    phoneme = ps[1][:-1]
                    phonemes.append(self.phoneme2idx[phoneme])
                    tones.append(self.tone2idx[tone])
                    poses.append(self.pos2idx["E"])
                    prosodys.append(self.prosody2idx["#0"])
        tokens = np.stack([phonemes, tones, poses, prosodys], 0)  # (C, T)
        return tokens

    @staticmethod
    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def pinyin2phoneme(pinyin_str):
        """拼音转音素（声母和韵母）。所有轻声的声调都用5表示，儿化音标志是er5"""
        pinyin_str = pinyin_str.strip()
        pinyin_str = re.sub(r"([jqx])u", r"\1v", pinyin_str)
        pinyin_str = re.sub(r"yu", "v", pinyin_str)
        pinyin_str = re.sub(r"yi|y", "i", pinyin_str)
        pinyin_str = re.sub(r"wu|w", "u", pinyin_str)
        pinyin_str = re.sub(r"un", "uen", pinyin_str)
        pinyin_str = re.sub(r"iu", "iou", pinyin_str)
        pinyin_str = re.sub(r"ui", "uei", pinyin_str)
        pinyin_str = re.sub(r"([zcs])i", r"\1ii", pinyin_str)
        pinyin_str = re.sub(r"([zcs]h|r)i", r"\1iii", pinyin_str)
        pinyins = re.split(r"\s+", pinyin_str)
        pinyins_new = []
        for pinyin in pinyins:
            chunks = re.split(r"^(\^|ng\d*$|[zcs]h|[bpmfdtnlgkhjqxzcsryw])(?!\d)", pinyin)
            chunks = [chunk for chunk in chunks if chunk is not None and len(chunk) > 0]
            pinyins_new.append(" ".join(chunks))
        pinyin_str_new = " ".join(pinyins_new)
        return pinyin_str_new

    def to_text(self, sequence):
        result = ""
        for phoneme_id, tone_id, label_id, prosody_id in sequence.transpose(1, 0):
            phoneme = self.idx2phoneme[phoneme_id]
            tone = self.idx2tone[tone_id]
            label = self.idx2pos[label_id]
            prosody = self.idx2prosody[prosody_id]
            if phoneme == "pad":
                break
            result += f"{phoneme}/{tone}/{label}/{prosody} "
        result = result.strip()
        return result
