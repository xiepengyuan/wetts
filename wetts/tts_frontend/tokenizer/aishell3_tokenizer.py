# -*- coding: utf-8 -*-

import re
from typing import List, Optional

from wetts.tts_frontend.tokenizer.baker_tokenizer import BakerTokenizer


class Aishell3Tokenizer(BakerTokenizer):
    def __init__(
        self,
        phonemes: Optional[List[str]] = None,
        tones: Optional[List[str]] = None,
        poses: Optional[List[str]] = None,
        prosodys: Optional[List[str]] = None,
    ):
        super().__init__(phonemes, tones, poses, prosodys)

    def __call__(self, text):
        text = self.convert_to_baker(text)
        return self.parse(text)

    @staticmethod
    def convert_to_baker(text):
        text = re.sub(r"%", "#1", text)
        text = re.sub(r"\$", "#4 。", text)
        return text
