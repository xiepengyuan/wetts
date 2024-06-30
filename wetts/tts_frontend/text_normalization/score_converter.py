# -*- coding: utf-8 -*-

import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class ScoreConverter(object):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"(?:比分).*\b(\d+)([-：:])(\d+)\b")

    def convert(self, text):
        while True:
            chars = list(text)
            match_objs = list(self.pattern.finditer(text))
            if len(match_objs) == 0:
                break
            for i in reversed(match_objs):
                num1, num2 = i[1], i[3]
                chars[i.start(3):i.end(3)] = num_converter.value_reading(num2)
                chars[i.start(2):i.end(2)] = ["比"]
                chars[i.start(1):i.end(1)] = num_converter.value_reading(num1)
            text = "".join(chars)

        return text
