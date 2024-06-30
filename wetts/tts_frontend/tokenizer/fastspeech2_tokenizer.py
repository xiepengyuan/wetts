# -*- coding: utf-8 -*-

import re
import numpy as np

from wetts.tts_frontend.tokenizer.fastspeech2 import Text2Sequence, symbols


class Fastspeech2Tokenizer:
    def __init__(
        self,
        pinyin_lexicon_path="/NeMo/nemo/collections/tts/data/fastspeech2_multi/text/pinyin-lexicon-rr.txt"
    ):
        self.text2sequence = Text2Sequence(symbols.symbols, is_unique=True)
        self.pinyin_lexicon = {}
        with open(pinyin_lexicon_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                elems = line.split()
                pinyin = elems[0]
                syllable = " ".join(elems[1:])
                self.pinyin_lexicon[pinyin] = syllable

    def __call__(self, text, show_text=False):
        new_tokens = []
        tokens = re.split(r"\s+", text)
        tokens = [token for token in tokens if token]
        print(tokens)
        for token in tokens:
            if re.fullmatch(r"#\d|\W", token):
                if re.fullmatch(r"#3", token):
                    new_tokens.append("sp")
            else:  # pinyin
                new_tokens.append(self.pinyin_lexicon[token])
        new_text = " ".join(new_tokens)
        new_text = "{" + new_text + "}"
        new_text = np.array(self.text2sequence(new_text))
        return new_text
