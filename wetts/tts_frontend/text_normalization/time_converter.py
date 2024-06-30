# -*- coding: utf-8 -*-

import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class TimeConverter(object):
    def __init__(self):
        super().__init__()
        time_str = "(?:(\d+)[：:](\d+))(?:[：:](\d+))?"
        self.pattern = re.compile(time_str)
        self.pattern2 = re.compile(rf"{time_str}(\s*[-—]+\s*){time_str}")

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern2.finditer(text))
        for i in reversed(match_objs):
            chars[i.start(4):i.end(4)] = ["到"]
        text = "".join(chars)
        text = self.only_time_convert(text)
        return text

    def only_time_convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for i in reversed(match_objs):
            hour, minute, second = i[1], i[2], i[3]
            if second:
                second_chars = num_converter._value_reading(second)
                if second_chars:
                    second_chars.append("秒")
                chars[i.start(3) - 1: i.end(3)] = second_chars
            if minute:
                minute_chars = num_converter._value_reading(minute)
                if minute_chars:
                    minute_chars.append("分")
                chars[i.start(2): i.end(2)] = minute_chars
            if hour:
                chars[i.start(1): i.end(1) + 1] = num_converter.value_reading(hour) + [
                    "点"
                ]
        text = "".join(chars)
        return text
