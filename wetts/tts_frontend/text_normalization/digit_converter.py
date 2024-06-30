# -*- coding: utf-8 -*-

import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class DigitConverter(object):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"\d{13,}")  # 千亿以上，数码读法
        self.full_pattern = re.compile(r"^(\d)\1{2,}|([68])\2+|23{2,}|6{3,}8{3,}|0\d+$")
        self.patterns1 = (
            re.compile(r"(?<=[A-Za-z_])\d+|(?<!\d)6{3,}(?!\d)"),
            re.compile(r"(?<!\d)23{2,}\b"),
        )
        self.patterns2 = (
            re.compile(r"(?:编号)\D{,2}(\d+)(?:\D|$)"),
            re.compile(r"(\d+)折"),
        )

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for i in reversed(match_objs):
            num = i[0]
            chars[i.start(0): i.end(0)] = num_converter.digital_reading(num, yao=False)
        text = "".join(chars)

        for pattern in self.patterns1:
            chars = list(text)
            match_objs = list(pattern.finditer(text))
            for i in reversed(match_objs):
                num = i[0]
                chars[i.start(0): i.end(0)] = num_converter.digital_reading(num, yao=False)
            text = "".join(chars)

        for pattern in self.patterns2:
            chars = list(text)
            match_objs = list(pattern.finditer(text))
            for i in reversed(match_objs):
                num = i[1]
                chars[i.start(1): i.end(1)] = num_converter.digital_reading(num, yao=False)
            text = "".join(chars)

        chars = list(text)
        match_obj = self.full_pattern.fullmatch(text)
        if match_obj:
            num = match_obj[0]
            chars[match_obj.start(0): match_obj.end(0)] = num_converter.digital_reading(num, yao=False)
        text = "".join(chars)

        return text
