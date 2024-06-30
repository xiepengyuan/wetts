# -*- coding: utf-8 -*-

import re


class PunctConverter(object):
    def __init__(self):
        super().__init__()

    def pre_convert(self, text):
        text = re.sub(r"\.\.+", "…", text)
        # text = re.sub(r"\W*([，。！？!?])\W*", r"\1", text)
        text = re.sub(r"(?<=\w)[*×](?=\d)", "乘以", text)
        return text

    def convert(self, text):
        # text = re.sub(r"(?<=\D)-(?=\D)", "杠", text)
        text = re.sub(r"×", "乘以", text)
        text = re.sub(r"xx(?=[的哥])", "某某", text)
        text = re.sub(r"[=＝]", "等于", text)
        return text
