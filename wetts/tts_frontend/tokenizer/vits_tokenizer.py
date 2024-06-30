# -*- coding: utf-8 -*-

import re
import numpy as np


class VitsTokenizer:
    def __init__(
        self,
        pinyin_lexicon_path,
        phone_table_path
    ):
        self.pinyin_lexicon = {}
        with open(pinyin_lexicon_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                elems = line.split()
                pinyin = elems[0]
                syllable = " ".join(elems[1:])
                self.pinyin_lexicon[pinyin] = syllable

        self.phone_dict = {}
        with open(phone_table_path) as p_f:
            for line in p_f:
                phone_id = line.strip().split()
                self.phone_dict[phone_id[0]] = int(phone_id[1])

    def __call__(self, text, show_text=False):
        phoneme_text = self.to_phoneme_text(text)
        seq = [self.phone_dict[symbol] for symbol in phoneme_text.split()]
        return seq

    def to_phoneme_text(self, text):
        new_tokens = ["sil"]
        tokens = re.split(r"\s+", text)
        tokens = [token for token in tokens if token]
        print("tokens: ", tokens)
        for i, token in enumerate(tokens):
            if re.fullmatch(r"\W", token):  # 标点符号
                continue
            elif re.fullmatch(r"#\d", token):  # 韵律标记
                if new_tokens[-1] == "#0":
                    new_tokens[-1] = token
                else:
                    new_tokens.append(token)
            elif re.fullmatch(r"[A-Z]", token):  # 英文字母
                new_tokens.append(self.pinyin_lexicon[token])
                new_tokens.append("#2")
            else:  # 拼音
                new_tokens.append(self.pinyin_lexicon[token])
                new_tokens.append("#0")
        phoneme_text = " ".join(new_tokens)
        print("phoneme_text: ", phoneme_text)
        return phoneme_text
