# -*- coding: utf-8 -*-

import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class DateConverter(object):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"(?<!\d)(?:(\d{2}|\d{4})年)(?:(\d{1,2})月(?:(\d{1,2})[日号])?)?")
        self.pattern1 = re.compile(r"(?<!\d)(?:(\d{2}|\d{4})[-/])(?:(\d{1,2})[-/](?:(\d{1,2}))?)?(?!\d)")

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for i in reversed(match_objs):
            year, month, day = i[1], i[2], i[3]
            if day:
                chars[i.start(3):i.end(3)] = num_converter.value_reading(day)
            if month:
                chars[i.start(2):i.end(2)] = num_converter.value_reading(month)
            if year:
                chars[i.start(1):i.end(1)] = num_converter.digital_reading(year)
        text = "".join(chars)

        chars = list(text)
        match_objs = list(self.pattern1.finditer(text))
        for i in reversed(match_objs):
            year, month, day = i[1], i[2], i[3]
            if day:
                chars[i.start(3):i.end(3)] = num_converter.value_reading(day) + ["日"]
            if month:
                chars[i.start(2):i.end(2) + 1] = num_converter.value_reading(month) + ["月"]
            if year:
                chars[i.start(1):i.end(1) + 1] = num_converter.digital_reading(year) + ["年"]
        text = "".join(chars)

        return text
