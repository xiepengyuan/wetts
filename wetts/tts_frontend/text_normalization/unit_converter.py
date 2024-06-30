# -*- coding: utf-8 -*-

import re

Unit = [
    "平米",
]


class UnitConverter(object):
    def __init__(self):
        super().__init__()
        unit_str = "|".join(Unit)
        self.pattern = re.compile(fr"(/){unit_str}")

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for i in reversed(match_objs):
            chars[i.start(1): i.end(1)] = ["每"]
        text = "".join(chars)
        return text
