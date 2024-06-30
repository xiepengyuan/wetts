# -*- coding: utf-8 -*-

import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class ValueConverter(object):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"(?<!\d|.)(2)(?!\w)")

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for i in reversed(match_objs):
            num = i[1]
            chars[i.start(1):i.end(1)] = num_converter.value_reading(num, liang=False)
        text = "".join(chars)

        return text
